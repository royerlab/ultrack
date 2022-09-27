import logging
import shutil
import warnings
from pathlib import Path
from queue import Queue
from typing import Callable, Dict, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import sqlalchemy as sqla
from sqlalchemy.orm import Session
from tqdm import tqdm

from ultrack.config.dataconfig import DataConfig
from ultrack.core.database import NO_PARENT, NodeDB

LOG = logging.getLogger(__name__)


def spatial_drift(df: pd.DataFrame, lag: int = 1) -> pd.Series:
    """Helper function to compute the drift of a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Ordered dataframe with columns `t`, `z`, `y`, and `x`.
    lag : int, optional
        `t` lag, by default 1

    Returns
    -------
    pd.Series
        Drift values, invalid values are 0.
    """
    df = df.sort_values("t")
    drift = np.sqrt(
        np.square(df[["z", "y", "x"]] - df[["z", "y", "x"]].shift(periods=lag)).sum(
            axis=1
        )
    )
    drift.values[:lag] = 0.0
    return drift


def estimate_drift(df: pd.DataFrame, quantile: float = 0.99) -> float:
    """Compute a estimate of the tracks drift.

    Parameters
    ----------
    df : pd.DataFrame
        Tracks dataframe, must have `track_id` column.
    quantile : float, optional
        Drift quantile, by default 0.99

    Returns
    -------
    float
        Drift from the given quantile.
    """
    distances = df.groupby("track_id").apply(spatial_drift)
    robust_max_distance = np.quantile(distances, quantile)
    LOG.info(f"{quantile} quantile spatial drift distance of {robust_max_distance}")
    return robust_max_distance


def add_track_ids_to_forest(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `track_id` and `parent_track_id` columns to forest `df`.
    Each maximal path receveis a unique `track_id`.

    Parameters
    ----------
    df : pd.DataFrame
        Forest defined by the `parent_id` column and the dataframe indices.

    Returns
    -------
    pd.DataFrame
        Inplace modified input dataframe with additional columns.
    """
    forest = {
        parent_id: group.index.tolist() for parent_id, group in df.groupby("parent_id")
    }

    roots = df.index[df["parent_id"] == NO_PARENT]

    df["track_id"] = NO_PARENT
    df["parent_track_id"] = NO_PARENT

    track_id = 1
    for root in roots:
        queue = Queue()
        queue.put((root, NO_PARENT))

        while not queue.empty():
            node, parent_track_id = queue.get()

            while True:
                df.loc[node, "track_id"] = track_id
                df.loc[node, "parent_track_id"] = parent_track_id

                children = forest.get(node, [])
                if len(children) == 0:
                    # end of track
                    break

                elif len(children) == 1:
                    node = children[0]

                elif len(children) == 2:
                    queue.put((children[0], track_id))
                    queue.put((children[1], track_id))
                    break

                else:
                    raise RuntimeError(
                        f"Something is wrong. Found {len(children)} children when parsing tracks, expected 0, 1, or 2."
                    )

            track_id += 1

    return df


def tracks_forest(df: pd.DataFrame) -> Dict[int, List[int]]:
    """
    Returns `track_id` and `parent_track_id` root-to-leaves forest (set of trees) graph structure.

    Example:
    forest[parent_id] = [child_id_0, child_id_1]
    """
    df = df.drop_duplicates(["track_id", "parent_track_id"])
    df = df[df["parent_track_id"] != NO_PARENT]
    graph = {}
    for parent_id, id in zip(df["parent_track_id"], df["track_id"]):
        graph[parent_id] = graph.get(parent_id, []) + [id]
    return graph


def inv_tracks_forest(df: pd.DataFrame) -> Dict[int, int]:
    """
    Returns `track_id` and `parent_track_id` leaves-to-root inverted forest (set of trees) graph structure.

    Example:
    forest[child_id] = parent_id
    """
    df = df.drop_duplicates(["track_id", "parent_track_id"])
    df = df[df["parent_track_id"] != NO_PARENT]
    graph = {}
    for parent_id, id in zip(df["parent_track_id"], df["track_id"]):
        graph[id] = parent_id
    return graph


def solution_dataframe_from_sql(
    database_path: str,
    columns: Sequence[sqla.Column] = (
        NodeDB.id,
        NodeDB.parent_id,
        NodeDB.t,
        NodeDB.z,
        NodeDB.y,
        NodeDB.x,
    ),
) -> pd.DataFrame:
    """Query `columns` of nodes in current solution (NodeDB.selected == True).

    Parameters
    ----------
    database_path : str
        SQL database path (e.g. sqlite:///your.database.db)

    columns : Sequence[sqla.Column], optional
        Queried columns, MUST include NodeDB.id.
        By default (NodeDB.id, NodeDB.parent_id, NodeDB.t, NodeDB.z, NodeDB.y, NodeDB.x)

    Returns
    -------
    pd.DataFrame
        Solution dataframe indexed by NodeDB.id
    """

    # query and convert tracking data to dataframe
    engine = sqla.create_engine(database_path)
    with Session(engine) as session:
        statement = session.query(*columns).where(NodeDB.selected).statement
        df = pd.read_sql(statement, session.bind, index_col="id")

    return df


def export_segmentation_generic(
    data_config: DataConfig,
    df: pd.DataFrame,
    export_func: Callable[[int, np.ndarray], None],
) -> None:
    """
    Generic function to export segmentation masks, segments labeled by `track_id` from `df`.

    Parameters
    ----------
    data_config : DataConfig
        Data parameters configuration.
    df : pd.DataFrame
        Tracks dataframe indexed by node id.
    export_func : Callable[[int, np.ndarray], None]
        Export function, it receives as input a time index `t` and its respective uint16 labeled buffer.
    """

    if "track_id" not in df.columns:
        raise ValueError(f"Dataframe must have `track_id` column. Found {df.columns}")

    LOG.info(f"Exporting segmentation masks with {export_func}")

    engine = sqla.create_engine(data_config.database_path)
    shape = data_config.metadata["shape"]

    df_indices = set(df.index)

    with Session(engine) as session:
        for t in tqdm(range(shape[0]), "Exporting segmentation masks"):
            buffer = np.zeros(shape[1:], dtype=np.uint16)
            query = list(
                session.query(NodeDB.id, NodeDB.pickle).where(
                    NodeDB.t == t, NodeDB.selected
                )
            )

            if len(query) == 0:
                warnings.warn(f"Segmentation mask from t = {t} is empty.")

            LOG.info(f"t = {t} containts {len(query)} segments.")

            for id, node in query:
                if id not in df_indices:
                    # ignoring nodes not present in dataset, used in sparse saving
                    # executing this with a sql query crashed with big datasets
                    continue

                track_id = df.loc[id, "track_id"]
                LOG.info(
                    f"Painting t = {t} node {id} with value {track_id} area {node.area}"
                )
                node.paint_buffer(buffer, value=track_id, include_time=False)

            export_func(t, buffer)


def large_chunk_size(
    shape: Tuple[int],
    dtype: Union[str, np.dtype],
    max_size: int = 2147483647,
) -> Tuple[int]:
    """
    Computes a large chunk size for a given `shape` and `dtype`.
    Large chunks improves the performance on Elastic Storage Systems (ESS).
    Leading dimension (time) will always be chunked as 1.

    Parameters
    ----------
    shape : Tuple[int]
        Input data shape.
    dtype : Union[str, np.dtype]
        Input data type.
    max_size : int, optional
        Reference maximum size, by default 2147483647

    Returns
    -------
    Tuple[int]
        Suggested chunk size.
    """

    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)

    plane_shape = np.minimum(shape[-2:], 32768)

    if len(shape) == 3:
        chunks = (1, *plane_shape)
    elif len(shape) == 4:
        depth = min(max_size // (dtype.itemsize * np.prod(plane_shape)), shape[1])
        chunks = (1, depth, *plane_shape)
    else:
        raise NotImplementedError(
            f"Large chunk size only implemented for 2,3-D + time arrays. Found {len(shape) - 1} + time."
        )

    return chunks


def maybe_overwrite_path(path: Path, overwrite: bool) -> None:
    """Validates existance of path (or dir) and overwrites it if requested."""
    if path.exists():
        if overwrite:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        else:
            raise ValueError(
                f"{path} already exists. Set `--overwrite` option to overwrite it."
            )
