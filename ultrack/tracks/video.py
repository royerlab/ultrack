import random
import shutil
import warnings
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Generator, Literal, Optional, Sequence, Union

import dask.array as da
import napari
import numpy as np
import pandas as pd
from napari.components import Camera
from napari.layers import Layer
from numpy.typing import ArrayLike
from pandas.core.groupby import DataFrameGroupBy
from tqdm import tqdm

from ultrack.tracks import sort_trees_by_length, split_tracks_df_by_lineage

try:
    from napari_animation import Animation

except (ImportError, ModuleNotFoundError):
    warnings.warn(
        "To export tracks as videos, please install napari-animation: pip install napari-animation"
    )


def _disable_thumbnails(viewer: napari.Viewer) -> None:
    for l in viewer.layers:
        l._update_thumbnail = lambda *args, **kwargs: None


@contextmanager
def _block_refresh(viewer: napari.Viewer) -> Generator[None, None, None]:
    """
    Disable layer refresh during context block.
    Useful to speed up state update when creating animation

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer.
    """
    prev_refresh = [l.refresh for l in viewer.layers]

    for l in viewer.layers:
        l.refresh = lambda *args, **kwargs: None

    yield

    for l, v in zip(viewer.layers, prev_refresh):
        l.refresh = v

    # NOTE: after napari 0.5.0 current solution can be replaced with:
    # prev_values = [
    #     l._refresh_blocked
    #     for l in viewer.layers
    # ]

    # for l in viewer.layers:
    #     l._refresh_blocked = True

    # yield

    # for l, v in zip(viewer.layers, prev_values):
    #     l._refresh_blocked = v


def _optimize_layers(viewer: napari.Viewer) -> None:
    """
    Convert layers to data types used by vispy.
    """
    size_threshold = 2048 * 2048 * 512
    for l in viewer.layers:
        if isinstance(l.data, da.Array) or l.data.size <= size_threshold:
            try:
                if l.data.dtype == np.uint16 or l.data.dtype == np.int16:
                    l.data = l.data.astype(np.int32)
                elif l.data.dtype == np.float16:
                    l.data = l.data.astype(np.float32)
            except Exception:
                pass


def record_tracks(
    viewer: napari.Viewer,
    tracks_df: pd.DataFrame,
    output_directory: Path,
    overwrite: bool,
    orthogonal_proj: bool,
    time_step: int,
    tracks_layer_kwargs: Dict[str, Any],
    tracks_layer_attrs: Dict[str, Any],
) -> None:
    """
    Record tracks as individual videos.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer with additional layers and configuration already setup.
    tracks_df : pd.DataFrame
        Tracks dataframe.
    output_directory : Path
        Output directory.
    overwrite : bool
        Whether to overwrite existing videos, otherwise they're skipped.
    orthogonal_proj : bool
        Whether to record orthogonal projections of entire tracks_df.
    time_step : int
        Napari animation keyframe time step.
    tracks_layer_kwargs : Dict[str, Any]
        Keyword arguments for the tracks layer creation.
    tracks_layer_attrs : Dict[str, Any]
        Keyword attributes set after the tracks layer is created.
    """
    root_id = int(tracks_df["track_id"].iloc[0])

    output_directory = output_directory / str(root_id)
    if output_directory.exists() and overwrite:
        shutil.rmtree(output_directory)

    output_directory.mkdir(exist_ok=True, parents=True)

    if "z" not in tracks_df.columns:
        cols = ["t", "y", "x"]
        ncols = 3
    else:
        cols = ["t", "z", "y", "x"]
        ncols = 4

    track_update_freq = 5

    # previous setup
    order = viewer.dims.order
    zoom = viewer.camera.zoom

    # ----- Single tracklet video ----- #
    viewer.dims.ndisplay = 2

    for axis_order in (0, 1):

        # converting to zx-slice
        if axis_order == 1:
            if ncols == 3:
                print("Skipping zx-slicing for 2D data.")
                continue
            new_order = order[:-3] + (order[-2], order[-3], order[-1])
            viewer.dims.order = new_order
            center_idx = [-3, -1]
        else:
            center_idx = [-2, -1]

        viewer.camera.zoom = zoom

        for track_id, tracklet in tracks_df.groupby("track_id", sort=True):
            out_path = output_directory / f"{track_id}_axis_{axis_order}_roi.mp4"
            if out_path.exists():
                print(f"Skipping {out_path}, it already exists.")
                continue

            tracks_layer = viewer.add_tracks(
                tracklet[["track_id", *cols]],
                **tracks_layer_kwargs,
            )
            for k, v in tracks_layer_attrs.items():
                setattr(tracks_layer, k, v)

            with _block_refresh(viewer):

                animation = Animation(viewer)

                for i, (_, s_track) in enumerate(
                    tracklet.groupby("t", sort=True, as_index=False)
                ):
                    if i % track_update_freq != 0:
                        continue

                    pos = s_track[cols].iloc[0].to_numpy()  # t, (z), y, x
                    viewer.dims.set_point(range(ncols), pos)
                    viewer.camera.center = pos[center_idx]
                    animation.capture_keyframe(track_update_freq * time_step)

                pos = s_track[cols].iloc[0].to_numpy()  # t, (z), y, x
                viewer.dims.set_point(range(ncols), pos)
                viewer.camera.center = pos[center_idx]
                animation.capture_keyframe((i % track_update_freq) * time_step)

            animation.animate(out_path, fps=60)

            viewer.layers.remove(tracks_layer)

    viewer.dims.order = order
    viewer.camera.zoom = zoom

    if not orthogonal_proj:
        return
    elif ncols == 3:
        print("Skipping orthogonal projections for 2D data.")
        return

    # ----- projection videos ----- #
    tracks_layer = viewer.add_tracks(
        tracks_df[["track_id", *cols]], **tracks_layer_kwargs
    )
    for k, v in tracks_layer_attrs.items():
        setattr(tracks_layer, k, v)

    viewer.dims.ndisplay = 3
    camera_angles = [(0, 0, 90), (0, 0, 180), (-90, 90, 0)]

    for proj_idx, angles in enumerate(camera_angles):

        out_path = output_directory / f"proj_{proj_idx}.mp4"
        if out_path.exists():
            print(f"Skipping {out_path}, it already exists.")
            continue

        t_min, t_max = tracks_df["t"].min(), tracks_df["t"].max()

        with _block_refresh(viewer):

            viewer.dims.set_point(0, t_min)
            viewer.camera.angles = angles

            animation = Animation(viewer)

            animation.capture_keyframe()

            viewer.dims.set_point(0, t_max)

            animation.capture_keyframe(int((t_max - t_min) * time_step))

        animation.animate(out_path, fps=60)

    # resetting
    viewer.layers.remove(tracks_layer)
    viewer.dims.order = order
    viewer.camera.zoom = zoom


def tracks_df_to_videos(
    viewer: napari.Viewer,
    tracks_df: pd.DataFrame,
    output_directory: Union[Path, str],
    num_lineages: Optional[int] = None,
    sort: Literal["random", "length", "id"] = "length",
    overwrite: bool = False,
    orthogonal_proj: bool = True,
    time_step: int = 5,
    tracks_layer_kwargs: Dict[str, Any] = {},
    tracks_layer_attrs: Dict[str, Any] = {},
) -> None:
    """
    Record tracks as individual videos.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer with additional layers and configuration already setup.
    tracks_df : pd.DataFrame
        Tracks dataframe, IMPORTANT: z, y, x should be at world (physical) coordinate system.
    output_directory : Path
        Output directory.
    num_lineages : int
        Number of lineages to record.
    sort : Literal["random", "length", "id"], optional
        Method to sort the lineages, by default "length".
    overwrite : bool, optional
        Whether to overwrite existing videos, otherwise they're skipped.
    orthogonal_proj : bool, optional
        Whether to record orthogonal projections of entire tracks_df.
    time_step : int, optional
        Napari animation keyframe time step.
    tracks_layer_kwargs : Dict[str, Any], optional
        Keyword arguments for the tracks layer creation.
    tracks_layer_attrs : Dict[str, Any], optional
        Keyword attributes set after the tracks layer is created.
    """

    if isinstance(output_directory, str):
        output_directory = Path(output_directory)

    output_directory.mkdir(exist_ok=True, parents=True)

    sort = sort.lower()

    if sort == "length":
        trees = sort_trees_by_length(tracks_df)
    else:
        trees = split_tracks_df_by_lineage(tracks_df)
        if sort == "random":
            random.shuffle(trees)
        elif sort != "id":
            raise ValueError(f"Unknown sort method: {sort}")

    if num_lineages is None:
        num_lineages = len(trees)

    if len(trees) < num_lineages:
        warnings.warn(
            f"Requested {num_lineages} lineages, but only {len(trees)} available."
        )

    _disable_thumbnails(viewer)
    _optimize_layers(viewer)

    for i, tree in tqdm(enumerate(trees), total=num_lineages):
        if i >= num_lineages:
            break

        record_tracks(
            viewer=viewer,
            tracks_df=tree,
            overwrite=overwrite,
            output_directory=output_directory,
            orthogonal_proj=orthogonal_proj,
            time_step=time_step,
            tracks_layer_kwargs=tracks_layer_kwargs,
            tracks_layer_attrs=tracks_layer_attrs,
        )


def _update_center_planes(
    viewer: napari.Viewer,
    clipping_planes_layers: Sequence[Layer],
    box_size: int,
    position: ArrayLike,
) -> None:
    """
    Update clipping planes to center around a position.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer.
    clipping_planes_layers : Sequence[Layer]
        Layers to be updated with clipping planes.
    box_size : int
        Size of the box around the position.
    position : ArrayLike
        3D position to center the box around.
    """

    if not clipping_planes_layers:
        return

    if len(position) != 3:
        raise ValueError(f"Position should be 3D. Found {len(position)} dimensions.")

    position = np.asarray(position)

    half_size = box_size // 2
    planes = []
    for i in range(3):
        for d in (-1, 1):
            p = position.copy()
            p[i] -= d * half_size
            n = np.zeros(3)
            n[i] = d
            planes.append({"normal": n, "position": p})

    n = np.asarray(viewer.camera.view_direction)
    planes.append({"normal": n, "position": position + 1})
    planes.append({"normal": -n, "position": position - 1})

    for l in clipping_planes_layers:
        l.experimental_clipping_planes = planes


def _linear_interp(
    current_t: int,
    start_t: int,
    end_t: Optional[int],
    start_val: ArrayLike,
    end_val: Optional[ArrayLike],
) -> ArrayLike:
    """
    Linear interpolation between two values.

    Parameters
    ----------
    current_t : int
        Current time.
    start_t : int
        Start time.
    end_t : int
        End time.
    start_val : ArrayLike
        Start value.
    end_val : ArrayLike
        End value.

    Returns
    -------
    ArrayLike
        Interpolated value.
    """
    if end_t is None:
        return start_val
    w = (current_t - start_t) / (end_t - start_t)
    w = np.clip(w, 0, 1)
    return start_val * (1 - w) + end_val * w


@dataclass
class CameraAttr:
    prev_t: int
    prev: ArrayLike
    next: Optional[ArrayLike]
    next_t: Optional[int]
    next_camera_attr: Optional["CameraAttr"] = None

    def current(self, t: int) -> ArrayLike:
        return _linear_interp(t, self.prev_t, self.next_t, self.prev, self.next)

    def maybe_next(self, t: int) -> "CameraAttr":
        if self.next_t is None or t < self.next_t:
            return self
        return self.next_camera_attr

    def create_next(self, t: int, val: int) -> "CameraAttr":
        self.next_t = t
        self.next = val
        self.next_camera_attr = CameraAttr(t, val, None, None)
        return self.next_camera_attr


def _maybe_get_auto_value(
    tracks_by_t: DataFrameGroupBy,
    t: int,
    key: str,
    value: ArrayLike,
    origin: ArrayLike,
) -> ArrayLike:
    """
    Get auto value for camera attributes.

    Parameters
    ----------
    tracks_by_t : DataFrameGroupBy
        Tracks dataframe grouped by time.
    t : int
        Time point.
    key : str
        Camera attribute key.
    value : ArrayLike
        Value to check.
    origin : ArrayLike
        Origin position.

    Returns
    -------
    ArrayLike
        Updated value.
    """
    if value == "auto":
        if key == "center":
            value = tracks_by_t.get_group(t)[["z", "y", "x"]].mean().to_numpy()
        elif key == "angles":
            point = tracks_by_t.get_group(t)[["z", "y", "x"]].mean().to_numpy()
            dummy_camera = Camera()
            dummy_camera.set_view_direction(origin - point)
            value = dummy_camera.angles
        else:
            raise ValueError(
                f"Only 'center' and 'angles' can be set to 'auto'. Found in '{key}'."
            )

    return np.asarray(value)


def _setup_camera_attrs(
    tracks_by_t: DataFrameGroupBy,
    cam: Camera,
    t_min: int,
    time_to_camera_kwargs: Dict[int, Dict[str, Any]],
    origin: ArrayLike,
) -> Dict[int, CameraAttr]:
    """
    Setup dictionary of linked lists for camera attributes for each time point.

    Parameters
    ----------
    tracks_by_t : DataFrameGroupBy
        Tracks dataframe grouped by time.
    cam : napari.components.Camera
        Napari camera.
    t_min : int
        Minimum time point.
    time_to_camera_kwargs : Dict[int, Dict[str, Any]]
        Dictionary of time points to camera attributes.
    origin : ArrayLike
        Origin position.

    Returns
    -------
    Dict[int, CameraAttr]
        Dictionary of camera attributes for each time point.
    """
    cam_keys = ("zoom", "angles", "center")
    for t, cam_kwargs in time_to_camera_kwargs.items():
        for k in cam_kwargs:
            if k not in cam_keys:
                raise ValueError(f"Unknown camera attribute '{k}' at key '{t}'.")

    if t_min not in time_to_camera_kwargs:
        time_to_camera_kwargs[t_min] = {}

    cam_attrs: Dict[int, CameraAttr] = {}
    for c in cam_keys:
        value = time_to_camera_kwargs[t_min].get(c, getattr(cam, c))

        value = _maybe_get_auto_value(tracks_by_t, t_min, c, value, origin)

        cam_attrs[c] = CameraAttr(t_min, value, None, None)

    tmp_cam_attrs = cam_attrs.copy()

    sorted_cam_t = list(sorted(time_to_camera_kwargs.keys()))
    for t in sorted_cam_t:
        if t == t_min:
            continue

        for k, v in time_to_camera_kwargs[t].items():
            v = _maybe_get_auto_value(tracks_by_t, t, k, v, origin)
            tmp_cam_attrs[k] = tmp_cam_attrs[k].create_next(t, v)

    return cam_attrs


def tracks_df_to_3D_video(
    viewer: napari.Viewer,
    tracks_df: pd.DataFrame,
    output_path: Union[Path, str],
    keyframe_step: int = 5,
    update_step: int = 5,
    clipping_planes_layers: Sequence[Layer] = (),
    clipping_box_size: int = 25,
    time_to_camera_kwargs: Dict[int, Dict[str, Any]] = {},
    time_to_viewer_cb: Dict[int, Callable[[napari.Viewer], None]] = {},
    origin: Optional[ArrayLike] = None,
    overwrite: bool = False,
) -> None:
    """
    Record 3D video of a single lineage with optional moving camera and clipping planes.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer with additional layers and configuration already setup.
        IMPORTANT: Including the tracks layer.
    tracks_df : pd.DataFrame
        Tracks dataframe, IMPORTANT: z, y, x should be at world (physical) coordinate system.
    output_path : Path
        Output path for the video.
    keyframe_step : int
        Keyframe step for napari animation, greater value creates longer movies.
    update_step : int
        Update time step, how often to update the viewer.
        The greater the values more often the positions will be updated.
    clipping_planes_layers : Sequence[Layer]
        Layers to be updated with clipping planes.
    clipping_box_size : int
        Size of the clipping plane bounding box around the position in world coordinates.
    time_to_camera_kwargs : Dict[int, Dict[str, Any]]
        Dictionary of time points to update camera attributes.
        Can include 'zoom', 'angles', 'center' per time point, for example:
        {0: {"zoom": 1.5, "angles": (0, 0, 90), "center": (0, 0, 0)}}
        "center" supports "auto" to set the center to the mean of the tracks at that time point.
    time_to_viewer_cb : Dict[int, Callable[[napari.Viewer]]]
        Dictionary of time points to viewer callback functions.
        Callbacks are called at the provided time point.
        Useful to update viewer settings, for example:
            ``{0: lambda viewer: viewer.dims.ndisplay = 3}``
    origin : ArrayLike
        Volume origin position, used to automatically set camera angles.
        When not provided, the mean of the tracks is used.
    overwrite : bool
        Whether to overwrite existing video.
    """

    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output path '{output_path}' already exists.")

    if not output_path.name.endswith(".mp4"):
        raise ValueError(f"Output path should be a .mp4 file. Found '{output_path}'.")

    tracks_df["t"] = tracks_df["t"].astype(int)
    t_min = tracks_df["t"].min().item()

    cols = ["z", "y", "x"]
    crange = tuple(range(1, 4))

    cam = viewer.camera
    tracks_by_t = tracks_df.groupby("t")

    if origin is None:
        origin = tracks_df[cols].mean().to_numpy()

    cam_attrs = _setup_camera_attrs(
        tracks_by_t,
        cam,
        t_min,
        time_to_camera_kwargs,
        origin=origin,
    )

    _disable_thumbnails(viewer)

    with _block_refresh(viewer):
        _optimize_layers(viewer)

        animation = Animation(viewer)

        for t in range(t_min, int(tracks_df["t"].max().item()) + 1):

            if t in time_to_viewer_cb:
                time_to_viewer_cb[t](viewer)

            if t != t_min and t % update_step != 0:
                continue

            for k, c in list(cam_attrs.items()):
                value = c.current(t)
                setattr(cam, k, value)
                cam_attrs[k] = c.maybe_next(t)

            center = tracks_by_t.get_group(t)[cols].mean().to_numpy()

            viewer.dims.set_point(0, t)
            viewer.dims.set_point(crange, center)
            _update_center_planes(
                viewer,
                clipping_planes_layers,
                clipping_box_size,
                center,
            )

            animation.capture_keyframe(update_step * keyframe_step)

        animation.capture_keyframe((t % update_step) * keyframe_step)

    animation.animate(output_path, fps=60)


def tracks_df_to_moving_2D_plane_video(
    viewer: napari.Viewer,
    tracks_df: pd.DataFrame,
    output_path: Union[Path, str],
    plane_mov_scale: float = 1.0,
    keyframe_step: int = 5,
    tracks_layer_kwargs: Dict[str, Any] = {},
    overwrite: bool = False,
) -> None:
    """
    Record 2D video of a single lineage with moving z-plane.

    Parameters
    ----------
    viewer : napari.Viewer
        Napari viewer with additional layers and configuration already setup.
    tracks_df : pd.DataFrame
        Tracks dataframe.
    output_path : Path
        Output path for the video.
    plane_mov_scale : float
        Scale factor for the moving image scale.
    keyframe_step : int
        Keyframe step for napari animation, greater value creates longer movies.
    tracks_layer_kwargs : Dict[str, Any]
        Keyword arguments for the tracks layer creation.
    overwrite : bool
        Whether to overwrite existing video.
    """
    output_path = Path(output_path)
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output path '{output_path}' already exists.")

    tracks_df["z"] = tracks_df["t"] * plane_mov_scale

    if viewer.dims.ndisplay != 3:
        warnings.warn(
            "Setting viewer to 3D display. Set viewer.dims.ndisplay = 3 and select desired camera settings."
        )

    other_layers = list(viewer.layers)

    for l in other_layers:
        # expanding z
        scale = list(l.scale)
        scale.insert(1, plane_mov_scale)
        l.data = l.data[:, None]
        l.scale = scale

    viewer.add_tracks(
        tracks_df[["track_id", "t", "z", "y", "x"]],
        **tracks_layer_kwargs,
    )

    _disable_thumbnails(viewer)

    viewer.layers.move(-1, 0)

    with _block_refresh(viewer):
        _optimize_layers(viewer)

        animation = Animation(viewer)
        t_min, t_max = tracks_df["t"].min(), tracks_df["t"].max()

        viewer.dims.set_point(0, t_min)
        animation.capture_keyframe()

        delta_t = t_max - t_min
        viewer.dims.set_point(0, t_max)
        for l in other_layers:
            T = list(l.translate)
            T[1] += delta_t * plane_mov_scale
            l.translate = T

        animation.capture_keyframe(delta_t * keyframe_step)

    animation.animate(output_path, fps=60)
