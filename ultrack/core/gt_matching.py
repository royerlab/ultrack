import logging
from contextlib import nullcontext
from typing import Dict, Optional, Tuple, Union

import fasteners
import numpy as np
import pandas as pd
import sqlalchemy as sqla
from numpy.typing import ArrayLike
from skimage.measure import regionprops
from sqlalchemy.orm import Session
from toolz import curry

from ultrack.config.config import MainConfig
from ultrack.core.database import NO_PARENT, GTLinkDB, GTNodeDB, NodeDB
from ultrack.core.linking.processing import compute_spatial_neighbors
from ultrack.core.segmentation.node import Node
from ultrack.core.solve.sqlgtmatcher import SQLGTMatcher
from ultrack.tracks.stats import estimate_drift
from ultrack.utils.multiprocessing import (
    multiprocessing_apply,
    multiprocessing_sqlite_lock,
)

LOG = logging.getLogger(__name__)


@curry
def _match_ground_truth_frame(
    time: int,
    gt_labels: ArrayLike,
    config: MainConfig,
    scale: Optional[ArrayLike],
    write_lock: Optional[fasteners.InterProcessLock],
    segmentation_gt: bool,
) -> float:
    """
    Matches candidate hypotheses to ground-truth labels for a given time point.
    Segmentation hypotheses must be pre-computed.

    Parameters
    ----------
    time : int
        Time point to match.
    gt_labels : ArrayLike
        Ground-truth labels.
    config : MainConfig
        Configuration object.
    scale : Optional[ArrayLike]
        Scale of the data for distance computation.
    write_lock : Optional[fasteners.InterProcessLock]
        Lock for writing to the database.
    segmentation_gt : bool
        Wether the ground-truth labels are segmentation masks or points.

    Returns
    -------
    float
        Mean score of the matching.
    """
    gt_labels = np.asarray(gt_labels[time])
    gt_props = regionprops(gt_labels)

    if len(gt_props) == 0:
        LOG.warning(f"No objects found in time point {time}")
        return

    LOG.info(f"Found {len(gt_props)} objects in time point {time}")

    gt_db_rows = []
    gt_nodes = []
    # adding ground-truth nodes
    for obj in gt_props:
        node = Node.from_mask(
            node_id=obj.label, time=time, mask=obj.image, bbox=obj.bbox
        )

        if len(node.centroid) == 2:
            y, x = node.centroid
            z = 0
        else:
            z, y, x = node.centroid

        gt_db_rows.append(
            GTNodeDB(
                t=time,
                label=obj.label,
                pickle=node,
                z=z,
                y=y,
                x=x,
            )
        )
        gt_nodes.append(node)

    with write_lock if write_lock is not None else nullcontext():
        connect_args = {"timeout": 45}
        engine = sqla.create_engine(
            config.data_config.database_path, connect_args=connect_args
        )

        with Session(engine) as session:
            session.add_all(gt_db_rows)
            session.commit()

            source_nodes = [
                n for n, in session.query(NodeDB.pickle).where(NodeDB.t == time)
            ]

        engine.dispose()

    if segmentation_gt:

        def _weight_func(tgt, src):
            return tgt.IoU(src)

    else:

        def _weight_func(tgt, src):
            return src.area * tgt.IoU(src)

    compute_spatial_neighbors(
        time,
        config=config.linking_config,
        source_nodes=source_nodes,
        target_nodes=gt_nodes,
        target_shift=np.zeros((len(gt_nodes), 3), dtype=np.float32),
        table_name=GTLinkDB.__tablename__,
        db_path=config.data_config.database_path,
        scale=scale,
        images=[],
        write_lock=write_lock,
        weight_func=_weight_func,
    )

    # computing GT matching
    gt_matcher = SQLGTMatcher(config, write_lock=write_lock)
    total_score = gt_matcher(time=time)

    if len(gt_db_rows) > 0:
        mean_score = total_score / len(gt_db_rows)
    else:
        mean_score = 0.0

    LOG.info(f"time {time} total score: {total_score:0.4f}")
    LOG.info(f"time {time} mean score: {mean_score:0.4f}")

    return mean_score


def _get_nodes_df_with_matches(
    database_path: str,
    features: bool,
) -> pd.DataFrame:
    """
    Gets nodes data frame with matched ground-truth labels.

    Parameters
    ----------
    database_path : str
        Path to the database file.
    features : bool
        Whether to return additional features for automatic parameter tuning.

    Returns
    -------
    pd.DataFrame
        DataFrame with matched nodes.
    """
    connect_args = {"timeout": 45}
    engine = sqla.create_engine(database_path, connect_args=connect_args)

    if features:
        query_cols = [
            NodeDB.id,
            NodeDB.hier_parent_id,
            NodeDB.area,
            NodeDB.frontier,
            NodeDB.t,
            NodeDB.z,
            NodeDB.y,
            NodeDB.x,
        ]
    else:
        query_cols = [NodeDB.id]

    with Session(engine) as session:

        node_query = session.query(*query_cols)
        node_df = pd.read_sql(node_query.statement, session.bind, index_col="id")

        gt_edge_query = session.query(
            GTLinkDB.source_id,
            GTLinkDB.target_id,
        ).where(GTLinkDB.selected)
        gt_df = pd.read_sql(
            gt_edge_query.statement, session.bind, index_col="source_id"
        )
        gt_df.rename(
            columns={"target_id": "gt_track_id"},
            inplace=True,
        )

        LOG.info(f"Found {len(node_df)} nodes and {len(gt_df)} ground-truth links")

    node_df = node_df.join(gt_df)
    node_df["gt_track_id"] = node_df["gt_track_id"].fillna(NO_PARENT).astype(int)

    if features:
        # similar to persistence, see segmentation.processing.get_nodes_features

        children_df = node_df[node_df["hier_parent_id"] > 0]
        node_df.loc[children_df.index, "parent_frontier"] = node_df.loc[
            children_df["hier_parent_id"], "frontier"
        ].to_numpy()
        node_df.loc[node_df["parent_frontier"].isna(), "parent_frontier"] = node_df[
            "frontier"
        ].max()

    return node_df


def match_to_ground_truth(
    config: MainConfig,
    gt_labels: ArrayLike,
    scale: Optional[ArrayLike] = None,
    track_id_graph: Optional[Dict[int, int]] = None,
    segmentation_gt: bool = True,
    optimize_config: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, MainConfig]]:
    """
    Matches nodes to ground-truth labels returning additional features for automatic parameter tuning.

    Tolerances for optimal configuration based on ground-truth matches:
    * `max_distance` + 1.0
    * `min_area` * 0.95
    * `max_area` * 1.025
    * `min_frontier` - 0.025

    Parameters
    ----------
    config : MainConfig
        Configuration object.
    gt_labels : ArrayLike
        Ground-truth labels.
    scale : Optional[ArrayLike], optional
        Scale of the data for distance computation, by default None.
    track_id_graph : Optional[Dict[int, int]], optional
        Ground-truth graph of track IDs, by default None.
    segmentation_gt : bool, optional
        Whether the ground-truth labels are segmentation masks or points, by default True.
    optimize_config : bool, optional
        Whether to find optimal configuration based on the ground-truth matches, by default False.
        If True, it will return the configuration object with updated parameters.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, MainConfig]]
        Data frame containing matched ground-truth labels to their respective nodes.
        If `optimize_config` is True, it will return a tuple with the data frame and the updated configuration object.
    """

    with multiprocessing_sqlite_lock(config.data_config) as lock:
        ious = multiprocessing_apply(
            _match_ground_truth_frame(
                gt_labels=gt_labels,
                config=config,
                scale=scale,
                write_lock=lock,
                segmentation_gt=segmentation_gt,
            ),
            range(gt_labels.shape[0]),
            n_workers=config.segmentation_config.n_workers,
            desc="Matching hierarchy nodes with ground-truth",
        )
        mean_iou = np.nanmean(ious)

        df_nodes = _get_nodes_df_with_matches(
            config.data_config.database_path,
            features=optimize_config,
        )

    if track_id_graph is not None:
        df_nodes["gt_parent_track_id"] = df_nodes["gt_track_id"].apply(
            lambda x: track_id_graph.get(x, NO_PARENT)
        )
    else:
        df_nodes["gt_parent_track_id"] = NO_PARENT

    if not optimize_config:
        return df_nodes

    # optimize configuration
    opt_config = config.copy(deep=True)
    gt_df = df_nodes[df_nodes["gt_track_id"] > 0]

    if len(gt_df) == 0:
        LOG.warning("No ground-truth matches found. Keeping previous configuration.")
        return df_nodes, opt_config

    if segmentation_gt:
        print(f"GT matching mean IoU (per frame): {mean_iou}")
        if mean_iou < 0.5:
            LOG.warning(
                "Mean IoU (per frame) is below 0.5. "
                "Bad matching between ground-truth and candidate segmentation hypotheses.\n"
                "Improve `foreground`, `contours` or relax initial configuration."
            )

    if scale is not None:
        cols = ["z", "y", "x"][-len(scale) :]
        gt_df[cols] *= scale

    if "z" not in gt_df.columns:
        gt_df["z"] = 0.0

    max_distance = estimate_drift(gt_df)
    if not np.isnan(max_distance) and max_distance > 0:
        opt_config.linking_config.max_distance = max_distance + 1.0

    opt_config.segmentation_config.min_area = gt_df["area"].min() * 0.95
    opt_config.segmentation_config.max_area = gt_df["area"].max() * 1.025

    opt_config.segmentation_config.min_frontier = max(
        gt_df["parent_frontier"].min() - 0.025, 0.0
    )

    return df_nodes, opt_config
