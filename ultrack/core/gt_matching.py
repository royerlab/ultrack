import logging
from contextlib import nullcontext
from typing import Optional

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
) -> None:
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
        engine = sqla.create_engine(config.data_config.database_path)

        with Session(engine) as session:
            session.add_all(gt_db_rows)
            session.commit()

            source_nodes = [
                n for n, in session.query(NodeDB.pickle).where(NodeDB.t == time)
            ]

        engine.dispose()

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


def _get_nodes_df_with_matches(database_path: str) -> pd.DataFrame:
    """
    Gets nodes data frame with matched ground-truth labels.

    Parameters
    ----------
    database_path : str
        Path to the database file.

    Returns
    -------
    pd.DataFrame
        DataFrame with matched nodes.
    """
    engine = sqla.create_engine(database_path)

    with Session(engine) as session:
        node_query = session.query(
            NodeDB.id,
            NodeDB.hier_parent_id,
            NodeDB.t_hier_id,
            # NodeDB.area,
            # NodeDB.frontier,
        )
        node_df = pd.read_sql(node_query.statement, session.bind, index_col="id")

        gt_edge_query = (
            session.query(
                GTLinkDB.source_id,
                GTLinkDB.target_id,
                # GTNodeDB.z,
                # GTNodeDB.y,
                # GTNodeDB.x,
            ).where(GTLinkDB.selected)
            # .join(GTNodeDB, GTNodeDB.id == GTLinkDB.target_id)
        )
        gt_df = pd.read_sql(
            gt_edge_query.statement, session.bind, index_col="source_id"
        )
        gt_df.rename(
            columns={
                "target_id": "gt_track_id"
            },  # , "z": "gt_z", "y": "gt_y", "x": "gt_x"},
            inplace=True,
        )

        LOG.info(f"Found {len(node_df)} nodes and {len(gt_df)} ground-truth links")

    node_df = node_df.join(gt_df)
    node_df["gt_track_id"] = node_df["gt_track_id"].fillna(NO_PARENT).astype(int)

    # frontiers = node_df["frontier"]
    # node_df["parent_frontier"] = node_df["hier_parent_id"].map(
    #     lambda x: frontiers.get(x, -1.0)
    # )
    # node_df.loc[node_df["parent_frontier"] < 0, "parent_frontier"] = node_df[
    #     "frontier"
    # ].max()

    return node_df


def match_to_ground_truth(
    config: MainConfig,
    gt_labels: ArrayLike,
    scale: Optional[ArrayLike] = None,
) -> pd.DataFrame:
    """
    Matches nodes to ground-truth labels returning additional features for automatic parameter tuning.

    Parameters
    ----------
    config : MainConfig
        Configuration object.
    gt_labels : ArrayLike
        Ground-truth labels.
    scale : Optional[ArrayLike], optional
        Scale of the data for distance computation, by default None.

    Returns
    -------
    pd.DataFrame
        Data frame containing matched ground-truth labels to their respective nodes.
    """

    with multiprocessing_sqlite_lock(config.data_config) as lock:
        multiprocessing_apply(
            _match_ground_truth_frame(
                gt_labels=gt_labels,
                config=config,
                scale=scale,
                write_lock=lock,
            ),
            range(gt_labels.shape[0]),
            n_workers=config.segmentation_config.n_workers,
            desc="Matching hierarchy nodes with ground-truth",
        )

        df_nodes = _get_nodes_df_with_matches(config.data_config.database_path)

    return df_nodes

    # if scale is not None:
    #     cols = ["z", "y", "x"][-len(scale) :]
    #     gt_df[cols] *= scale

    # if "z" not in gt_df.columns:
    #     gt_df["z"] = 0.0

    # if len(gt_df) > 0:
    #     max_distance = estimate_drift(gt_df)
    #     if not np.isnan(max_distance) or max_distance > 0:
    #         config.linking_config.max_distance = max_distance + 1.0

    # if "solution" in df.columns:
    #     matched_df = df[df["solution"] > 0.5]
    #     config.segmentation_config.min_area = matched_df["area"].min() * 0.95

    #     config.segmentation_config.max_area = matched_df["area"].max() * 1.025

    #     config.segmentation_config.min_frontier = max(
    #         matched_df["parent_frontier"].min() - 0.025, 0.0
    #     )
    # else:
    #     LOG.warning("No nodes were matched. Keeping previous configuration.")

    # config.data_config.database = prev_db

    # return config, df
