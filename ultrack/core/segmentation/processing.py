import logging
import pickle
from contextlib import nullcontext
from typing import Optional

import fasteners
import numpy as np
import pandas as pd
import sqlalchemy as sqla
from numpy.typing import ArrayLike
from toolz import curry

from ultrack.config import DataConfig, SegmentationConfig
from ultrack.core.database import NO_PARENT, Base, NodeDB, OverlapDB
from ultrack.core.segmentation.hierarchy import create_hierarchies
from ultrack.core.segmentation.utils import check_array_chunk
from ultrack.utils.multiprocessing import (
    multiprocessing_apply,
    multiprocessing_sqlite_lock,
)

logging.basicConfig()
logging.getLogger("sqlachemy.engine").setLevel(logging.INFO)

LOG = logging.getLogger(__name__)


def generate_id(index: int, time: int, max_segments: int) -> int:
    """Generates a unique id.

    Parameters
    ----------
    index : int
        Current index per time.
    time_point : int
        Current time.
    max_segments : int
        Upper bound of number of segments.

    Returns
    -------
    int
        Unique id.
    """
    return index + (time + 1) * max_segments


@curry
def _process(
    time: int,
    detection: ArrayLike,
    edge: ArrayLike,
    config: SegmentationConfig,
    db_path: str,
    max_segments_per_time: int,
    lock: Optional[fasteners.InterProcessLock] = None,
) -> None:
    """Process `detection` and `edge` of current time and add data to database.

    Parameters
    ----------
    time : int
        Current time.
    detection : ArrayLike
        Detection array of current `time`.
    edge : ArrayLike
        Edge array of current `time`.
    config : InitConfig
        Initialization configuration parameters.
    db_path : str
        Path to database including type prefix.
    max_segments_per_time : int
        Upper bound of segments per time point.
    lock : Optional[fasteners.Lock], optional
        Lock object for SQLite multiprocessing, optional otherwise, by default None.
    """
    np.random.seed(42)  # necessary because of watershed tie-zones

    hiers = create_hierarchies(
        detection[time] > config.threshold,
        edge[time],
        hierarchy_fun=config.ws_hierarchy,
        max_area=config.max_area,
        min_area=config.min_area,
        min_frontier=config.min_frontier,
    )

    LOG.info(f"Computing nodes of time {time}")

    nodes = []
    overlap_source = []
    overlap_ancestor = []
    index = 1
    for h, hierarchy in enumerate(hiers):
        hierarchy.cache = True

        hier_index_map = {}
        for node in hierarchy.nodes:
            node.id = generate_id(index, time, max_segments_per_time)
            node.time = time
            node._parent = None  # avoiding pickling parent hierarchy
            centroid = node.centroid

            hier_index_map[node._h_node_index] = node.id

            if len(centroid) == 2:
                y, x = centroid
                z = 0
            else:
                z, y, x = centroid

            nodes.append(
                {
                    "id": node.id,
                    "t_node_id": index,
                    "t_hier_id": h + 1,
                    "t": time,
                    "z": z,
                    "y": y,
                    "x": x,
                    "selected": False,
                    "area": node.area,
                    "parent_id": NO_PARENT,
                    "pickle": pickle.dumps(node),
                }
            )
            index += 1

        tree = hierarchy.tree
        # find overlapping segments by iterating through hierarchy (tree)
        for h_index in hier_index_map.keys():
            overlaps = [
                hier_index_map[a]
                for a in tree.ancestors(h_index)
                if a in hier_index_map and a != h_index  # roots point to it self
            ]
            overlap_source += [hier_index_map[h_index]] * len(overlaps)
            overlap_ancestor += overlaps
            LOG.info(
                f"{len(overlaps)} overlaps found for node {hier_index_map[h_index]}."
            )

        hierarchy.cache = False
        hierarchy.free_props()

        if index > max_segments_per_time:
            raise ValueError(
                f"Number of segments exceeds upper bound of {max_segments_per_time} per time."
            )

    nodes = pd.DataFrame(nodes)
    overlaps = pd.DataFrame(
        np.asarray([overlap_source, overlap_ancestor]).T,
        columns=["node_id", "ancestor_id"],
    )

    with lock if lock is not None else nullcontext():
        LOG.info(f"Pushing nodes from time {time} to {db_path}")
        engine = sqla.create_engine(db_path, hide_parameters=True)
        with engine.begin() as conn:
            nodes.to_sql(
                name=NodeDB.__tablename__, con=conn, if_exists="append", index=False
            )
            overlaps.to_sql(
                name=OverlapDB.__tablename__, con=conn, if_exists="append", index=False
            )


def segment(
    detection: ArrayLike,
    edge: ArrayLike,
    segmentation_config: SegmentationConfig,
    data_config: DataConfig,
    max_segments_per_time: int = 1_000_000,
) -> None:
    """Add candidate segmentation (nodes) from `detection` and `edge` to database.

    Parameters
    ----------
    detection : ArrayLike
        Fuzzy detection array of shape (T, (Z), Y, X)
    edge : ArrayLike
        Edge array of shape (T, (Z), Y, X)
    segmentation_config : SegmentationConfig
        Segmentation configuration parameters.
    data_config : DataConfig
        Data configuration parameters.
    max_segments_per_time : int
        Upper bound of segments per time point.
    """
    LOG.info(f"Adding nodes with SegmentationConfig:\n{segmentation_config}")

    if detection.shape != edge.shape:
        raise ValueError(
            f"`detection` and `edge` shape must match. Found {detection.shape} and {edge.shape}"
        )

    check_array_chunk(detection)
    check_array_chunk(edge)

    LOG.info(f"Detection array with shape {detection.shape}")
    LOG.info(f"Edge array with shape {edge.shape}")

    engine = sqla.create_engine(data_config.database_path)
    Base.metadata.create_all(engine)

    with multiprocessing_sqlite_lock(data_config) as lock:

        process = _process(
            detection=detection,
            edge=edge,
            config=segmentation_config,
            db_path=data_config.database_path,
            lock=lock,
            max_segments_per_time=max_segments_per_time,
        )

        length = detection.shape[0]
        multiprocessing_apply(
            process,
            range(length),
            segmentation_config.n_workers,
            desc="Adding nodes to database",
        )
