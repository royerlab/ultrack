import logging
from contextlib import nullcontext
from typing import List, Optional, Sequence

import fasteners
import numpy as np
import pandas as pd
import sqlalchemy as sqla
from numpy.typing import ArrayLike
from scipy.spatial import KDTree
from sqlalchemy.orm import Session
from toolz import curry

from ultrack.config import DataConfig, LinkingConfig
from ultrack.core.database import LinkDB, NodeDB, maximum_time
from ultrack.core.linking.utils import clear_linking_data
from ultrack.core.segmentation.node import Node
from ultrack.utils.multiprocessing import (
    batch_index_range,
    multiprocessing_apply,
    multiprocessing_sqlite_lock,
)

logging.basicConfig()
logging.getLogger("sqlachemy.engine").setLevel(logging.INFO)

LOG = logging.getLogger(__name__)


def _compute_dct(
    time: int,
    nodes: List[Node],
    images: Sequence[ArrayLike],
) -> None:
    """Precomputes DCT values for the nodes using the frames from the provided time."""

    frames = [image[time] for image in images]
    LOG.info(f"Image with shape {[f.shape for f in frames]}")

    for node in nodes:
        node.precompute_dct(frames)


@curry
def _process(
    time: int,
    config: LinkingConfig,
    db_path: str,
    images: Sequence[ArrayLike],
    write_lock: Optional[fasteners.InterProcessLock] = None,
) -> None:
    """Link nodes from current time to time + 1.

    Parameters
    ----------
    time : int
        Current time.
    config : LinkingConfig
        Linking configuration parameters.
    db_path : str
        Database path.
    images : Sequence[ArrayLike]
        Sequence of images for DCT correlation edge weight, if empty, IoU is used for weighting.
    write_lock : Optional[fasteners.InterProcessLock], optional
        Lock object for SQLite multiprocessing, optional otherwise, by default None.
    """
    engine = sqla.create_engine(db_path)
    with Session(engine) as session:
        query = session.query(NodeDB.pickle)

        current_nodes = [n for n, in query.where(NodeDB.t == time)]
        next_nodes = [n for n, in query.where(NodeDB.t == time + 1)]

    current_pos = np.asarray([n.centroid for n in current_nodes])
    next_pos = np.asarray([n.centroid for n in next_nodes])

    # finds neighbors nodes within the radius
    # and connect the pairs with highest edge weight
    current_kdtree = KDTree(current_pos)

    distances, neighbors = current_kdtree.query(
        next_pos,
        # twice as expected because we select the nearest with highest edge weight
        k=2 * config.max_neighbors,
        distance_upper_bound=config.max_distance,
    )

    if len(images) > 0:
        LOG.info("DCT edge weight")
        LOG.info(f"computing DCT of nodes from t={time}")
        _compute_dct(time, current_nodes, images)
        _compute_dct(time + 1, next_nodes, images)
        weight_func = Node.dct_dot
    else:
        LOG.info("IoU edge weight")
        weight_func = Node.IoU

    distance_w = config.distance_weight
    links = []

    for i, node in enumerate(next_nodes):
        valid = np.logical_not(np.isinf(distances[i]))
        valid_neighbors = neighbors[i, valid]
        neigh_distances = distances[i, valid]

        neighborhood = []
        for neigh_idx, neigh_dist in zip(valid_neighbors, neigh_distances):
            neigh = current_nodes[neigh_idx]
            edge_weight = weight_func(node, neigh) - distance_w * neigh_dist
            # using dist as a tie-breaker
            neighborhood.append(
                (edge_weight, -neigh_dist, neigh.id, node.id)
            )  # current, next

        neighborhood = sorted(neighborhood, reverse=True)[: config.max_neighbors]
        LOG.info(f"Node {node.id} links {neighborhood}")
        links += neighborhood

    links = np.asarray(links)[:, [0, 2, 3]]  # ignoring index column
    df = pd.DataFrame(links, columns=["iou", "source_id", "target_id"])

    with write_lock if write_lock is not None else nullcontext():
        LOG.info(f"Pushing links from time {time} to {db_path}")
        engine = sqla.create_engine(db_path, hide_parameters=True)
        with engine.begin() as conn:
            df.to_sql(
                name=LinkDB.__tablename__, con=conn, if_exists="append", index=False
            )


def link(
    linking_config: LinkingConfig,
    data_config: DataConfig,
    images: Sequence[ArrayLike] = tuple(),
    batch_index: Optional[int] = None,
    overwrite: bool = False,
) -> None:
    """Links candidate segments (nodes) with their neighbors on the next time.

    Parameters
    ----------
    linking_config : LinkingConfig
        Linking configuration parameters.
    data_config : DataConfig
        Data configuration parameters.
    images : Sequence[ArrayLike]
        Optinal sequence of images for DCT correlation edge weight.
    batch_index : Optional[int], optional
        Batch index for processing a subset of nodes, by default everything is processed.
    overwrite : bool
        Cleans up linking database content before processing.
    """
    LOG.info(f"Linking nodes with LinkingConfig:\n{linking_config}")

    max_t = maximum_time(data_config)
    time_points = batch_index_range(max_t, linking_config.n_workers, batch_index)
    LOG.info(f"Linking time points {time_points}")

    if overwrite and (batch_index is None or batch_index == 0):
        clear_linking_data(data_config.database_path)

    with multiprocessing_sqlite_lock(data_config) as lock:
        process = _process(
            config=linking_config,
            db_path=data_config.database_path,
            write_lock=lock,
            images=images,
        )
        multiprocessing_apply(
            process, time_points, linking_config.n_workers, desc="Linking nodes."
        )
