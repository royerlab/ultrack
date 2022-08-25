import logging
from queue import Queue
from typing import Dict, List, Sequence

import numpy as np
import pandas as pd
import sqlalchemy as sqla
from sqlalchemy.orm import Session

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
