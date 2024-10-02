import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import sqlalchemy as sqla
from numpy.typing import ArrayLike
from scipy.ndimage import generate_binary_structure, grey_dilation, zoom
from scipy.optimize import linear_sum_assignment
from scipy.spatial import KDTree
from scipy.spatial.distance import cdist
from skimage.measure import regionprops
from skimage.segmentation import relabel_sequential
from sqlalchemy.orm import Session
from tifffile import imwrite
from toolz import curry
from tqdm import tqdm

from ultrack.config.config import MainConfig
from ultrack.config.dataconfig import DataConfig
from ultrack.core.database import NodeDB
from ultrack.core.export.utils import (
    export_segmentation_generic,
    filter_nodes_generic,
    solution_dataframe_from_sql,
)
from ultrack.core.segmentation.node import Node, intersects
from ultrack.tracks.graph import (
    add_track_ids_to_tracks_df,
    get_subtree,
    tracks_df_forest,
)
from ultrack.tracks.stats import estimate_drift
from ultrack.utils.constants import NO_PARENT
from ultrack.utils.data import validate_and_overwrite_path

logging.basicConfig()
logging.getLogger("sqlachemy.engine").setLevel(logging.INFO)

LOG = logging.getLogger(__name__)


def ctc_compress_forest(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compress each maximal path into the cell-tracking challenge format.
    Reference: https://public.celltrackingchallenge.net/documents/Naming%20and%20file%20content%20conventions.pdf

    It contains 4 columns L, B, E, P, where:
        - L is unique track label;
        - B is a zero-based track time begin;
        - E is a zero-based track time end;
        - P is parent track index, 0 when parent-less.

    Parameters
    ----------
    df : pd.DataFrame
        Forest dataframe with `track_id` and `parent_track_id` columns.

    Returns
    -------
    pd.DataFrame
        Compressed path forest.
    """
    ctc_df = []
    for track_id, group in df.groupby("track_id"):
        ctc_df.append(
            {
                "L": track_id,
                "B": int(group["t"].min()),
                "E": int(group["t"].max()),
                "P": group["parent_track_id"].iloc[0],
            }
        )

    ctc_df = pd.DataFrame(ctc_df)
    ctc_df.loc[ctc_df["P"] == NO_PARENT, "P"] = 0

    return ctc_df


def stitch_tracks_df(
    graph: Dict[int, List[int]],
    df: pd.DataFrame,
    selected_track_ids: Set[int],
) -> pd.DataFrame:
    """Filters selected tracks and stitches (connects) incomplete tracks to nearby tracks on subsequent time point.

    Parameters
    ----------
    graph : Dict[int, List[int]]
        Forest graph indexed by subtree root.
    df : pd.DataFrame
        Tracks dataframe.
    selected_track_ids : Set[int]
        Set of selected tracks to be kept.

    Returns
    -------
    pd.DataFrame
        Filtered and stitched tracks dataframe.
    """
    selected_track_ids = selected_track_ids.copy()

    start_df = []
    end_df = []
    track_ids = []
    for i, group in df.groupby("track_id", as_index=True):
        start_df.append(group.loc[group["t"].idxmin()])
        end_df.append(group.loc[group["t"].idxmax()])
        track_ids.append(int(i))

    start_df = pd.DataFrame(start_df, index=track_ids)
    # removing samples from divisions
    start_df = start_df[start_df["parent_track_id"] == NO_PARENT]

    end_df = pd.DataFrame(end_df, index=track_ids)
    # removing samples from divisions
    end_df = end_df[np.logical_not(end_df.index.isin(df["parent_track_id"]))]

    start_by_t = start_df.groupby("t")
    track_id_mapping = {i: i for i in track_ids}
    track_id_mapping[NO_PARENT] = NO_PARENT

    max_distance = estimate_drift(df)

    for t, end_group in end_df.groupby("t"):
        end_group = end_group[end_group.index.isin(selected_track_ids)]

        LOG.info(f"# {len(end_group)} track ends at t = {t}")
        if len(end_group) == 0:
            continue

        try:
            start_group = start_by_t.get_group(t + 1)
        except KeyError:
            continue

        start_group = start_group[
            np.logical_not(start_group.index.isin(selected_track_ids))
        ]

        LOG.info(f"# {len(end_group)} track ends at t = {t + 1}")
        if len(start_group) == 0:
            continue

        kdtree = KDTree(start_group[["z", "y", "x"]])
        dist, neighbors = kdtree.query(
            end_group[["z", "y", "x"]],
            distance_upper_bound=max_distance,
        )

        # removing invalid neighbors
        valid = dist != np.inf
        neighbors = neighbors[valid]
        neigh_df = pd.DataFrame(
            {"dist": dist[valid], "neighbor": start_group.index.values[neighbors]},
            index=end_group.index.values[valid],
        )

        neigh_df = neigh_df.sort_values(by=["dist"])
        neigh_df = neigh_df.drop_duplicates("neighbor", keep="first")

        LOG.info(
            f"stitching {neigh_df.index.values} to {neigh_df['neighbor'].values} at t = {t}"
        )

        for prev_id, new_id in zip(neigh_df["neighbor"], neigh_df.index):
            track_id_mapping[int(prev_id)] = int(new_id)
            selected_track_ids.update(get_subtree(graph, prev_id))

    LOG.info(f"track_id remapping {track_id_mapping}")

    df["track_id"] = df["track_id"].map(track_id_mapping)
    df["parent_track_id"] = df["parent_track_id"].map(track_id_mapping)
    df = df[df["track_id"].isin(selected_track_ids)]

    return df


def select_tracks_from_first_frame(
    data_config: DataConfig,
    first_frame: pd.DataFrame,
    df: pd.DataFrame,
    stitch_tracks: bool,
) -> None:
    """
    Selects only the subset of tracks rooted at the detection masks of `first_frame` array.

    Parameters
    ----------
    data_config : DataConfig
        Data configuration parameters.
    first_frame : pd.DataFrame
        Detection mask of first frame.
    df : pd.DataFrame
        Tracks dataframe.
    stitch_tracks : bool
        Stitches (connects) incomplete tracks nearby tracks on subsequent time point.
    """

    # query starting nodes data
    engine = sqla.create_engine(data_config.database_path)
    with Session(engine) as session:
        query = session.query(NodeDB.pickle).where(NodeDB.t == 0, NodeDB.selected)
        starting_nodes = [n for n, in query]

    root_centroids = np.asarray([n.centroid for n in starting_nodes])
    marker_centroids = np.asarray(
        [n.centroid for n in regionprops(first_frame, cache=False)]
    )
    D = cdist(marker_centroids, root_centroids)

    _, root_ids = linear_sum_assignment(D)

    selected_track_ids = set()
    graph = tracks_df_forest(df)

    for root in tqdm(root_ids, "Selecting tracks from first trame"):
        track_id = df.loc[starting_nodes[root].id, "track_id"]
        selected_track_ids.update(get_subtree(graph, track_id))

    if stitch_tracks:
        selected_df = stitch_tracks_df(graph, df, selected_track_ids)
    else:
        selected_df = df[df["track_id"].isin(selected_track_ids)].copy()

    return selected_df


def _validate_masks_path(output_dir: Path, overwrite: bool) -> None:
    """Validates existance of output masks paths."""
    mask_paths = list(output_dir.glob("mask*.tif"))
    if len(mask_paths) > 0:
        if overwrite:
            for path in mask_paths:
                path.unlink()
        else:
            raise ValueError(
                f"{output_dir}'s segmentation masks on already exists. Set `--overwrite` option to overwrite it."
            )


@curry
def _write_tiff_buffer(
    t: int,
    buffer: np.ndarray,
    output_dir: Path,
    scale: Optional[ArrayLike] = None,
    dilation_iters: int = 0,
) -> None:
    """Writes a single tiff stack into `output_dir` / "mask%03d.tif"

    Parameters
    ----------
    t : int
        Time index.
    buffer : np.ndarray
        Segmentation mask uint16 buffer.
    output_dir : Path
        Output directory.
    scale : Optional[ArrayLike], optional
        Mask rescaling factor, by default None
    dilation_iters: int
        Iterations of radius 1 morphological dilations on labels, applied after scaling, by default 0.
    """
    if scale is not None:
        buffer = zoom(
            buffer, scale[-buffer.ndim :], order=0, grid_mode=True, mode="grid-constant"
        )

    footprint = generate_binary_structure(buffer.ndim, 1)
    for _ in range(dilation_iters):
        dilated = grey_dilation(buffer, footprint=footprint)
        np.putmask(buffer, buffer == 0, dilated)

    # reducing mask sizes
    buffer = buffer.astype(
        np.promote_types(np.min_scalar_type(buffer.max()), np.uint16)
    )

    imwrite(output_dir / f"mask{t:03}.tif", buffer, compression="LZW")


def to_ctc(
    output_dir: Path,
    config: MainConfig,
    margin: int = 0,
    scale: Optional[Tuple[float]] = None,
    first_frame: Optional[ArrayLike] = None,
    dilation_iters: int = 0,
    stitch_tracks: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Exports tracking results to cell-tracking challenge (http://celltrackingchallenge.net) format.

    Parameters
    ----------
    output_dir : Path
        Output directory to save segmentation masks and lineage graph
    config : DataConfig
        Configuration parameters.
    scale : Optional[Tuple[float]], optional
        Optional scaling of output segmentation masks, by default None
    margin : int
        Margin used to filter out nodes and splitting their tracklets
    first_frame : Optional[ArrayLike], optional
        Optional first frame detection mask to select a subset of tracks (e.g. Fluo-N3DL-DRO), by default None
    dilation_iters: int
        Iterations of radius 1 morphological dilations on labels, applied after scaling, by default 0.
    stitch_tracks: bool, optional
        Stitches (connects) incomplete tracks nearby tracks on subsequent time point, by default False
    overwrite : bool, optional
        Flag to overwrite existing `output_dir` content, by default False
    """
    if stitch_tracks and first_frame is None:
        raise NotImplementedError(
            "Tracks stitching only implemented for when `first_frame` is supplied."
        )

    output_dir.mkdir(exist_ok=True, parents=True)
    tracks_path = output_dir / "res_track.txt"

    _validate_masks_path(output_dir, overwrite)
    validate_and_overwrite_path(tracks_path, overwrite, "cli")

    df = solution_dataframe_from_sql(config.data_config.database_path)

    if len(df) == 0:
        raise ValueError("Solution is empty.")

    condition = None
    if scale is not None and not np.all(np.isclose(scale, 1.0)):
        condition = rescale_size_condition(scale)

    if margin > 0:
        if condition is None:
            condition = margin_filter_condition(config.data_config, margin)
        else:
            # mergin two conditions into one to avoid looping querying the db twice
            first_cond = condition
            second_cond = margin_filter_condition(config.data_config, margin)

            def condition(node: Node) -> bool:
                return first_cond(node) or second_cond(node)

    if condition is not None:
        df = filter_nodes_generic(config.data_config, df, condition)

    if len(df) == 0:
        raise ValueError("Solution is empty after filtering.")

    df = add_track_ids_to_tracks_df(df)

    if first_frame is not None:
        if scale is not None:
            first_frame = zoom(
                first_frame,
                1 / np.asarray(scale)[-first_frame.ndim :],
                order=0,
                grid_mode=True,
                mode="grid-constant",
            )

        df = select_tracks_from_first_frame(
            config.data_config,
            first_frame,
            df,
            stitch_tracks=stitch_tracks,
        )

    df["track_id"], fw, _ = relabel_sequential(df["track_id"].values)
    fw[NO_PARENT] = NO_PARENT
    df["parent_track_id"] = fw[df["parent_track_id"].values]

    # convert to CTC format and write output
    tracks_df = ctc_compress_forest(df)
    tracks_df.to_csv(tracks_path, sep=" ", header=False, index=False)

    LOG.info(f"CTC tracking data:\n{tracks_df}")

    export_segmentation_generic(
        config.data_config,
        df,
        _write_tiff_buffer(
            output_dir=output_dir, scale=scale, dilation_iters=dilation_iters
        ),
    )


def margin_filter_condition(
    data_config: DataConfig,
    margin: int,
) -> Callable[[Node], bool]:
    """Condition to check if nodes are completely inside margin.

    Parameters
    ----------
    data_config : DataConfig
        Data configuration parameters.
    margin : int
        yx-axis margin.

    Returns
    -------
    Callable[[Node], bool]
        Condition checking if node is completely inside the margins.
    """
    # ignoring time
    upper_lim = np.asarray(data_config.metadata["shape"][1:])
    upper_lim[-2:] -= margin  # only for y, x coordinates

    lower_lim = np.zeros_like(upper_lim)
    lower_lim[-2:] += margin  # only for y, x coordinates

    limits = np.concatenate((lower_lim, upper_lim))

    LOG.info(f"Using limits {limits} from {margin}")

    def _condition(node: Node) -> bool:
        return not intersects(node.bbox, limits)

    return _condition


def rescale_size_condition(
    scale: ArrayLike,
) -> Callable[[Node], bool]:
    """Condition to check if a node will disappear after scaling.

    Parameters
    ----------
    scale : ArrayLike
        Scaling factors.

    Returns
    -------
    Callable[[Node], bool]
        Condition to check if node is valid after scaling.
    """
    scale = np.asarray(scale)

    def _condition(node: Node) -> bool:
        ndim = node.mask.ndim
        return np.any(node.mask.shape * scale[-ndim:] < 1.0)

    return _condition
