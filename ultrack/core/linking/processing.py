import logging
from contextlib import nullcontext
from typing import Callable, List, Optional, Sequence

import fasteners
import numpy as np
import pandas as pd
import sqlalchemy as sqla
from numpy.typing import ArrayLike
from scipy.spatial import KDTree
from sqlalchemy.orm import Session
from toolz import curry

from ultrack.config.config import LinkingConfig, MainConfig
from ultrack.core.database import LinkDB, NodeDB, maximum_time_from_database
from ultrack.core.linking.utils import clear_linking_data
from ultrack.core.segmentation.node import Node
from ultrack.utils.array import check_array_chunk
from ultrack.utils.multiprocessing import (
    batch_index_range,
    multiprocessing_apply,
    multiprocessing_sqlite_lock,
)

logging.basicConfig()
logging.getLogger("sqlachemy.engine").setLevel(logging.INFO)

LOG = logging.getLogger(__name__)


def _compute_features(
    time: int,
    nodes: List[Node],
    images: Sequence[ArrayLike],
    feature_funcs: List[Callable[[Node, ArrayLike], ArrayLike]],
) -> List[ArrayLike]:
    """Compute mean intensity for each node"""

    frames = np.stack(
        [np.asarray(image[time]) for image in images],
        axis=-1,
    )
    LOG.info(f"Image with shape {[f.shape for f in frames]}")

    return [
        np.stack([func(node, frames) for node in nodes], axis=0)
        for func in feature_funcs
    ]


def color_filtering_mask(
    time: int,
    current_nodes: List[Node],
    next_nodes: List[Node],
    images: Sequence[ArrayLike],
    neighbors: ArrayLike,
    z_score_threshold: float,
) -> ArrayLike:
    """
    Filtering by color z-score.

    Parameters
    ----------
    time : int
        Current time.
    current_nodes : List[Node]
        List of source nodes.
    next_nodes : List[Node]
        List of target nodes.
    images : Sequence[ArrayLike]
        Sequence of images to extract color features for filtering.
    neighbors : ArrayLike
        Neighbors indices (current/source) for each target (next) node.
    z_score_threshold : float
        Z-score threshold for color filtering.

    Returns
    -------
    ArrayLike
        Boolean mask of neighboring nodes within color z-score threshold.

    """
    LOG.info(f"computing filtering by color z-score from t={time}")
    (current_features,) = _compute_features(
        time, current_nodes, images, [Node.intensity_mean]
    )
    # inserting dummy value for missing neighbors
    current_features = np.append(
        current_features,
        np.zeros((1, current_features.shape[1])),
        axis=0,
    )
    next_features, next_features_std = _compute_features(
        time + 1, next_nodes, images, [Node.intensity_mean, Node.intensity_std]
    )
    LOG.info(
        f"Features Std. Dev. range {next_features_std.min()} {next_features_std.max()}"
    )
    next_features_std[next_features_std <= 1e-6] = 1.0
    difference = next_features[:, None, ...] - current_features[neighbors]
    difference /= next_features_std[:, None, ...]
    filtered_by_color = np.abs(difference).max(axis=-1) <= z_score_threshold
    return filtered_by_color


@curry
def _process(
    time: int,
    config: LinkingConfig,
    db_path: str,
    images: Sequence[ArrayLike],
    scale: Optional[Sequence[float]],
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
        Sequence of images for color space filtering, if empty, no filtering is performed.
    scale : Sequence[float]
        Optional scaling for nodes' distances.
    write_lock : Optional[fasteners.InterProcessLock], optional
        Lock object for SQLite multiprocessing, optional otherwise, by default None.
    """
    connect_args = {"timeout": 45} if write_lock is not None else {}
    engine = sqla.create_engine(db_path, connect_args=connect_args)
    with Session(engine) as session:
        current_nodes = [
            n for n, in session.query(NodeDB.pickle).where(NodeDB.t == time)
        ]

        query = session.query(
            NodeDB.pickle,
            NodeDB.z_shift,
            NodeDB.y_shift,
            NodeDB.x_shift,
        ).where(NodeDB.t == time + 1)

        next_nodes = [row[0] for row in query]
        next_shift = np.asarray([row[1:] for row in query])

    compute_spatial_neighbors(
        time,
        config,
        current_nodes,
        next_nodes,
        next_shift,
        scale=scale,
        table_name=LinkDB.__tablename__,
        db_path=db_path,
        images=images,
        write_lock=write_lock,
    )


def compute_spatial_neighbors(
    time: int,
    config: LinkingConfig,
    source_nodes: List[Node],
    target_nodes: List[Node],
    target_shift: ArrayLike,
    scale: Optional[Sequence[float]],
    table_name: str,
    db_path: str,
    images: Sequence[ArrayLike],
    write_lock: Optional[fasteners.InterProcessLock] = None,
    weight_func: Callable[[Node, Node], float] = Node.IoU,
) -> pd.DataFrame:

    source_pos = np.asarray([n.centroid for n in source_nodes])
    target_pos = np.asarray([n.centroid for n in target_nodes], dtype=np.float32)

    n_dim = target_pos.shape[1]
    target_shift = target_shift[:, -n_dim:]  # matching positions dimensions
    target_pos += target_shift

    if scale is not None:
        min_n_dim = min(n_dim, len(scale))
        scale = scale[-min_n_dim:]
        source_pos = source_pos[..., -min_n_dim:] * scale
        target_pos = target_pos[..., -min_n_dim:] * scale

    # finds neighbors nodes within the radius
    # and connect the pairs with highest edge weight
    current_kdtree = KDTree(source_pos)

    distances, neighbors = current_kdtree.query(
        target_pos,
        # twice as expected because we select the nearest with highest edge weight
        k=2 * config.max_neighbors,
        distance_upper_bound=config.max_distance,
    )

    if len(images) > 0:
        filtered_by_color = color_filtering_mask(
            time,
            source_nodes,
            target_nodes,
            images,
            neighbors,
            config.z_score_threshold,
        )
    else:
        filtered_by_color = np.ones_like(neighbors, dtype=bool)

    int_next_shift = np.round(target_shift).astype(int)
    # NOTE: moving bbox with shift, MUST be after `feature computation`
    for node, shift in zip(target_nodes, int_next_shift):
        node.bbox[:n_dim] += shift
        node.bbox[-n_dim:] += shift

    distance_w = config.distance_weight
    links = []

    for i, node in enumerate(target_nodes):
        valid = (~np.isinf(distances[i])) & filtered_by_color[i]
        valid_neighbors = neighbors[i, valid]
        neigh_distances = distances[i, valid]

        neighborhood = []
        for neigh_idx, neigh_dist in zip(valid_neighbors, neigh_distances):
            neigh = source_nodes[neigh_idx]
            edge_weight = weight_func(node, neigh) - distance_w * neigh_dist
            # using dist as a tie-breaker
            neighborhood.append(
                (edge_weight, -neigh_dist, neigh.id, node.id)
            )  # current, next

        neighborhood = sorted(neighborhood, reverse=True)[: config.max_neighbors]
        LOG.info("Node %s links %s", node.id, neighborhood)
        links += neighborhood

    if len(links) == 0:
        raise ValueError(
            f"No links found for time {time}. Increase `linking_config.max_distance` parameter."
        )

    links = np.asarray(links)[:, [0, 2, 3]]  # ignoring index column
    df = pd.DataFrame(links, columns=["weight", "source_id", "target_id"])

    with write_lock if write_lock is not None else nullcontext():
        LOG.info(f"Pushing links from time {time} to {db_path}")
        connect_args = {"timeout": 45} if write_lock is not None else {}
        engine = sqla.create_engine(
            db_path, hide_parameters=True, connect_args=connect_args
        )
        with engine.begin() as conn:
            df.to_sql(name=table_name, con=conn, if_exists="append", index=False)

    return df


def link(
    config: MainConfig,
    images: Sequence[ArrayLike] = tuple(),
    scale: Optional[Sequence[float]] = None,
    batch_index: Optional[int] = None,
    overwrite: bool = False,
) -> None:
    """Links candidate segments (nodes) with their neighbors on the next time.

    Parameters
    ----------
    config : MainConfig
        Configuration parameters.
    images : Sequence[ArrayLike]
        Optinal sequence of images for color space filtering.
    scale : Sequence[float]
        Optional scaling for nodes' distances.
    batch_index : Optional[int], optional
        Batch index for processing a subset of nodes, by default everything is processed.
    overwrite : bool
        Cleans up linking database content before processing.
    """
    LOG.info(f"Linking nodes with LinkingConfig:\n{config.linking_config}")

    for image in images:
        check_array_chunk(image)

    max_t = maximum_time_from_database(config.data_config)
    time_points = batch_index_range(max_t, config.linking_config.n_workers, batch_index)
    LOG.info(f"Linking time points {time_points}")

    if overwrite and (batch_index is None or batch_index == 0):
        clear_linking_data(config.data_config.database_path)

    with multiprocessing_sqlite_lock(config.data_config) as lock:
        process = _process(
            config=config.linking_config,
            db_path=config.data_config.database_path,
            write_lock=lock,
            images=images,
            scale=scale,
        )
        multiprocessing_apply(
            process, time_points, config.linking_config.n_workers, desc="Linking nodes."
        )


def add_links(
    config: MainConfig,
    sources: ArrayLike,
    targets: ArrayLike,
    weights: ArrayLike,
) -> None:
    """
    Adds user-defined links to the database.

    Parameters
    ----------
    config : MainConfig
        Configuration parameters.
    sources : ArrayLike
        Sources (t) node id.
    targets : ArrayLike
        Targets (t + 1) node id.
    weights : ArrayLike
        Link weights, the higher the weight the more likely the link.
    """
    df = pd.DataFrame(
        {
            "source_id": np.asarray(sources, dtype=int),
            "target_id": np.asarray(targets, dtype=int),
            "weight": weights,
        }
    )

    engine = sqla.create_engine(
        config.data_config.database_path,
        hide_parameters=True,
    )

    with engine.begin() as conn:
        df.to_sql(name=LinkDB.__tablename__, con=conn, if_exists="append", index=False)
