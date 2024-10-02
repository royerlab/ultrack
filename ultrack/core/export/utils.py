import logging
import warnings
from typing import Callable, Sequence, Tuple

import numpy as np
import pandas as pd
import sqlalchemy as sqla
from sqlalchemy.orm import Session
from toolz import curry

from ultrack.config.dataconfig import DataConfig
from ultrack.core.database import NodeDB
from ultrack.core.segmentation.node import Node
from ultrack.utils.constants import NO_PARENT
from ultrack.utils.multiprocessing import multiprocessing_apply

LOG = logging.getLogger(__name__)


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
        buffer = np.zeros(shape, dtype=int)
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


@curry
def _filter_nodes_by_per_time_point(
    time: int,
    df: pd.DataFrame,
    database_path: str,
    condition: Callable[[Node], bool],
) -> pd.DataFrame:
    """Auxiliary function to filter nodes per time point.

    Parameters
    ----------
    time : int
        Time index.
    df : pd.DataFrame
        Tracks dataframe.
    database_path : str
        Nodes database path.
    condition : Callable[[Node], bool]
        Condition used to evaluate each node.

    Returns
    -------
    pd.DataFrame
        Filtered dataframe of the given time point.
    """
    df = df[df["t"] == time]
    removed_ids = set()

    engine = sqla.create_engine(database_path)
    with Session(engine) as session:
        nodes = session.query(NodeDB.pickle).where(NodeDB.id.in_(df.index))
        for (node,) in nodes:
            if condition(node):
                removed_ids.add(node.id)

    LOG.info(f"Removing nodes {removed_ids} at time {time}")

    df = df.drop(removed_ids)

    return df


def filter_nodes_generic(
    data_config: DataConfig,
    df: pd.DataFrame,
    condition: Callable[[Node], bool],
) -> pd.DataFrame:
    """Filter out nodes where `condition` is true.

    Parameters
    ----------
    data_config : DataConfig
        Data configuration parameters.
    df : pd.DataFrame
        Tracks dataframe.
    condition : Callable[[Node], bool]
        Condition used to evaluate each node.

    Returns
    -------
    pd.DataFrame
        Filtered tracks dataframe.
    """
    shape = data_config.metadata["shape"]

    df = pd.concat(
        multiprocessing_apply(
            _filter_nodes_by_per_time_point(
                database_path=data_config.database_path,
                df=df,
                condition=condition,
            ),
            sequence=range(shape[0]),
            n_workers=data_config.n_workers,
            desc="Filtering nodes",
        )
    )

    orphan = np.logical_not(df["parent_id"].isin(df.index))
    df.loc[orphan, "parent_id"] = NO_PARENT

    return df
