import random
import shutil
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Union

import napari
import pandas as pd
from tqdm import tqdm

from ultrack.tracks import sort_trees_by_length, split_tracks_df_by_lineage


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

    try:
        from napari_animation import Animation

    except ImportError as e:
        raise ImportError(
            "To export tracks as videos, please install napari-animation: pip install napari-animation"
        ) from e

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
            viewer.camera.zoom = zoom
            center_idx = [-3, -1]
        else:
            center_idx = [-2, -1]

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
        Tracks dataframe.
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
