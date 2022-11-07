import logging
import shutil
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import sqlalchemy as sqla
from numba import njit, typed, types
from sqlalchemy.orm import Session
from toolz import curry

from ultrack.config.dataconfig import DataConfig
from ultrack.core.database import NO_PARENT, NodeDB
from ultrack.utils.multiprocessing import multiprocessing_apply

LOG = logging.getLogger(__name__)


@njit
def _fast_path_transverse(
    node: int,
    track_id: int,
    queue: List[Tuple[int, int]],
    forest: Dict[int, Tuple[int]],
) -> List[int]:
    """Transverse a path in the forest directed graph and add path (track) split into queue.

    Parameters
    ----------
    node : int
        Source path node.
    track_id : int
        Reference track id for path split.
    queue : List[Tuple[int, int]]
        Source nodes and path (track) id reference queue.
    forest : Dict[int, Tuple[int]]
        Directed graph (tree) of paths relationships.

    Returns
    -------
    List[int]
        Sequence of nodes in the path.
    """
    path = typed.List.empty_list(types.int64)

    while True:
        path.append(node)

        children = forest.get(node)
        if children is None:
            # end of track
            break

        elif len(children) == 1:
            node = children[0]

        elif len(children) == 2:
            queue.append((children[1], track_id))
            queue.append((children[0], track_id))
            break

        else:
            raise RuntimeError(
                "Something is wrong. Found node with more than two children when parsing tracks."
            )

    return path


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
    df.index = df.index.astype(int)
    df["parent_id"] = df["parent_id"].astype(int)

    forest = typed.Dict.empty(types.int64, types.ListType(types.int64))
    for parent_id, group in df.groupby("parent_id"):
        forest[parent_id] = typed.List(group.index)
    roots = forest.pop(NO_PARENT)

    df["track_id"] = NO_PARENT
    df["parent_track_id"] = NO_PARENT

    track_id = 1
    for root in roots:
        queue = typed.List.empty_list(types.Tuple((types.int64, types.int64)))
        queue.append((root, NO_PARENT))

        while queue:
            node, parent_track_id = queue.pop()
            path = _fast_path_transverse(node, track_id, queue, forest)

            df.loc[path, "track_id"] = track_id
            df.loc[path, "parent_track_id"] = parent_track_id

            track_id += 1

    unlabeled_tracks = df["track_id"] == NO_PARENT
    assert not np.any(
        unlabeled_tracks
    ), f"Something went wrong. Found unlabeled tracks\n{df[unlabeled_tracks]}"

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


@curry
def _query_and_export_data_to_frame(
    time: int,
    database_path: str,
    shape: Tuple[int],
    df: pd.DataFrame,
    export_func: Callable[[int, np.ndarray], None],
) -> None:
    """Queries segmentation data from database and paints it according to their respective `df` `track_id` column.

    Parameters
    ----------
    time : int
        Frame time point to paint.
    database_path : str
        Database path.
    shape : Tuple[int]
        Frame shape.
    df : pd.DataFrame
        Tracks dataframe.
    export_func : Callable[[int, np.ndarray], None]
        Export function, it receives as input a time index `t` and its respective uint16 labeled buffer.
    """
    node_indices = set(df[df["t"] == time].index)

    engine = sqla.create_engine(database_path)
    with Session(engine) as session:
        buffer = np.zeros(shape, dtype=np.uint16)
        query = list(
            session.query(NodeDB.id, NodeDB.pickle).where(
                NodeDB.t == time, NodeDB.selected
            )
        )

        if len(query) == 0:
            warnings.warn(f"Segmentation mask from t = {time} is empty.")

        LOG.info(f"t = {time} containts {len(query)} segments.")

        for id, node in query:
            if id not in node_indices:
                # ignoring nodes not present in dataset, used when exporting a subset of data
                # filtering through a sql query crashed with big datasets
                continue

            track_id = df.loc[id, "track_id"]
            LOG.info(
                f"Painting t = {time} node {id} with value {track_id} area {node.area}"
            )
            node.paint_buffer(buffer, value=track_id, include_time=False)

        export_func(time, buffer)


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

    shape = data_config.metadata["shape"]

    multiprocessing_apply(
        _query_and_export_data_to_frame(
            database_path=data_config.database_path,
            shape=shape[1:],
            df=df,
            export_func=export_func,
        ),
        sequence=range(shape[0]),
        n_workers=data_config.n_workers,
        desc="Exporting segmentation masks",
    )


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
