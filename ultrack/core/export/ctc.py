import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import sqlalchemy as sqla
from numpy.typing import ArrayLike
from scipy.ndimage import zoom
from scipy.spatial import KDTree
from skimage.measure import regionprops
from sqlalchemy.orm import Session
from tifffile import imwrite
from tqdm import tqdm

from ultrack.config import DataConfig
from ultrack.core.database import NO_PARENT, NodeDB
from ultrack.core.export.utils import (
    add_track_ids_to_forest,
    estimate_drift,
    solution_dataframe_from_sql,
    tracks_forest,
)

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
            selected_track_ids.update(connected_component(graph, prev_id))

    LOG.info(f"track_id remapping {track_id_mapping}")

    df["track_id"] = df["track_id"].map(track_id_mapping)
    df["parent_track_id"] = df["parent_track_id"].map(track_id_mapping)
    df = df[df["track_id"].isin(selected_track_ids)]

    return df


def connected_component(graph: Dict[int, int], index: int) -> Set[int]:
    """Returns connected component (subtree) of `index` of tree `graph`."""
    component = set()
    queue = [index]
    while queue:
        index = queue.pop()
        component.add(index)
        for child in graph.get(index, []):
            queue.append(child)

    return component


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

    selected_track_ids = set()
    centroids = np.asarray([n.centroid for n in starting_nodes])

    graph = tracks_forest(df)

    for det in tqdm(regionprops(first_frame), "Selecting tracks from first trame"):
        # select nearest node that contains the reference detection.
        dist = np.square(centroids - det.centroid).sum(axis=1)
        node = starting_nodes[np.argmin(dist)]
        if node.contains(det.centroid):
            # add the whole tree to the selection
            track_id = df.loc[node.id, "track_id"]
            selected_track_ids.add(connected_component(graph, track_id))

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
                f"{output_dir}'s segmentation masks on already exists. Set `overwrite` option to overwrite it."
            )


def _validate_tracks_path(tracks_path: Path, overwrite: bool) -> None:
    """Validates existance of tracks path"""
    if tracks_path.exists():
        if overwrite:
            tracks_path.unlink()
        else:
            raise ValueError(
                f"{tracks_path} already exists. Set `overwrite` option to overwrite it."
            )


def to_ctc(
    output_dir: Path,
    data_config: DataConfig,
    scale: Optional[Tuple[float]] = None,
    first_frame: Optional[ArrayLike] = None,
    stitch_tracks: bool = False,
    overwrite: bool = False,
) -> None:
    """
    Exports tracking results to cell-tracking challenge (http://celltrackingchallenge.net) format.

    Parameters
    ----------
    output_dir : Path
        Output directory to save segmentation masks and lineage graph.
    data_config : DataConfig
        Data configuration parameters.
    scale : Optional[Tuple[float]], optional
        Optional scaling of output segmentation masks, by default None
    first_frame : Optional[ArrayLike], optional
        Optional first frame detection mask to select a subset of tracks (e.g. Fluo-N3DL-DRO), by default None
    stitch_tracks: bool, optional
        Stitches (connects) incomplete tracks nearby tracks on subsequent time point, by default False
    overwrite : bool, optional
        Flag to overwrite existing `output_dir` content, by default False
    """
    if stitch_tracks and first_frame is None:
        raise NotImplementedError(
            "Tracks stitching only implemented for when `first_frame` is supplied."
        )

    output_dir.mkdir(exist_ok=True)
    tracks_path = output_dir / "res_track.txt"

    _validate_masks_path(output_dir, overwrite)
    _validate_tracks_path(tracks_path, overwrite)

    df = solution_dataframe_from_sql(data_config.database_path)

    if len(df) == 0:
        raise ValueError("Solution is empty.")

    df = add_track_ids_to_forest(df)

    if first_frame is not None:
        if scale is not None:
            first_frame = zoom(
                first_frame, 1 / np.asarray(scale)[-first_frame.ndim :], order=0
            )

        df = select_tracks_from_first_frame(
            data_config, first_frame, df, stitch_tracks=False
        )

    # convert to CTC format and write output
    tracks_df = ctc_compress_forest(df)
    tracks_df.to_csv(tracks_path, sep=" ", header=False, index=False)

    LOG.info(f"CTC tracking data:\n{tracks_df}")

    shape = data_config.metadata["shape"]

    engine = sqla.create_engine(data_config.database_path)
    with Session(engine) as session:
        for t in tqdm(df["t"].unique(), "Saving CTC masks"):
            buffer = np.zeros(shape[1:], dtype=np.uint16)
            query = list(
                session.query(NodeDB.id, NodeDB.pickle).where(
                    NodeDB.t == t.item(), NodeDB.selected
                )
            )

            if len(query) == 0:
                warnings.warn(f"Segmentation mask from t = {t} is empty.")

            LOG.info(f"t = {t} containts {len(query)} segments.")

            for id, node in query:
                LOG.info(f"Painting t = {t} with node {id}.")

                node.paint_buffer(
                    buffer, value=df.loc[id, "track_id"], include_time=False
                )

            if scale is not None:
                buffer = zoom(buffer, scale[-buffer.ndim :], order=0)

            imwrite(output_dir / f"mask{t:03}.tif", buffer)
