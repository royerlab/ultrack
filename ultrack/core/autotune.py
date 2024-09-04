import logging
from contextlib import nullcontext
from typing import List, Literal, Optional, Tuple

import fasteners
import mip
import mip.exceptions
import numpy as np
import pandas as pd
import sqlalchemy as sqla
from numpy.typing import ArrayLike
from scipy.spatial import KDTree
from skimage.measure import regionprops
from sqlalchemy.orm import Session
from toolz import curry

from ultrack.config.config import MainConfig
from ultrack.core.database import GTLinkDB, GTNodeDB, NodeDB, OverlapDB
from ultrack.core.segmentation.node import Node
from ultrack.utils.multiprocessing import multiprocessing_apply

LOG = logging.getLogger(__name__)


def _link_gt(
    time: int,
    gt_nodes: List[Node],
    config: MainConfig,
    scale: Optional[ArrayLike],
    write_lock: Optional[fasteners.InterProcessLock],
) -> pd.DataFrame:

    if len(gt_nodes) == 0:
        LOG.warn(f"No ground-truth nodes found at {time}")
        return

    db_path = config.data_config.database_path

    engine = sqla.create_engine(db_path)
    with Session(engine) as session:
        h_nodes = [n for n, in session.query(NodeDB.id).where(NodeDB.t == time)]

    h_nodes_pos = np.asarray([n.centroids for n in h_nodes])
    gt_pos = np.asarray([n.centroids for n in gt_nodes])

    n_dim = h_nodes_pos.shape[-1]

    if scale is not None:
        min_n_dim = min(n_dim, len(scale))
        scale = scale[-min_n_dim:]
        h_nodes_pos = h_nodes_pos[..., -min_n_dim:] * scale
        gt_pos = gt_pos[..., -min_n_dim:] * scale

    # finds neighbors nodes within the radius
    # and connect the pairs with highest edge weight
    current_kdtree = KDTree(h_nodes_pos)

    distances, neighbors = current_kdtree.query(
        gt_pos,
        # twice as expected because we select the nearest with highest edge weight
        k=2 * config.linking_config.max_neighbors,
        distance_upper_bound=config.linking_config.max_distance,
    )

    gt_links = []

    for i, node in enumerate(gt_nodes):
        valid = ~np.isinf(distances[i])
        valid_neighbors = neighbors[i, valid]
        neigh_distances = distances[i, valid]

        neighborhood = []
        for neigh_idx, neigh_dist in zip(valid_neighbors, neigh_distances):
            neigh = h_nodes[neigh_idx]
            edge_weight = node.IoU(neigh)
            # using dist as a tie-breaker
            neighborhood.append(
                (edge_weight, -neigh_dist, neigh.id, node.id)
            )  # current, next

        neighborhood = sorted(neighborhood, reverse=True)[: config.max_neighbors]
        LOG.info("Node %s links %s", node.id, neighborhood)
        gt_links += neighborhood

    if len(gt_links) == 0:
        raise ValueError(
            f"No links found for time {time}. Increase `linking_config.max_distance` parameter."
        )

    gt_links = np.asarray(gt_links)[:, [0, 2, 3]]
    df = pd.DataFrame(gt_links, columns=["weight", "source_id", "target_id"])

    with write_lock if write_lock is not None else nullcontext():
        LOG.info(f"Pushing gt links from time {time} to {db_path}")
        engine = sqla.create_engine(
            db_path,
            hide_parameters=True,
        )
        with engine.begin() as conn:
            df.to_sql(
                name=GTLinkDB.__tablename__, con=conn, if_exists="append", index=False
            )

    return df


@curry
def _match_ground_truth_frame(
    time: int,
    gt_labels: ArrayLike,
    config: MainConfig,
    scale: Optional[ArrayLike],
    write_lock: Optional[fasteners.InterProcessLock],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # TODO

    gt_labels = np.asarray(gt_labels[time])
    gt_props = regionprops(gt_labels)

    if len(gt_props) == 0:
        LOG.warning(f"No objects found in time point {time}")

    gt_db_rows = []
    gt_nodes = []
    # adding ground-truth nodes
    for obj in gt_props:
        node = Node.from_mask(
            node_id=obj.label, time=time, mask=obj.image, bbox=obj.bbox
        )

        gt_db_rows.append(
            GTNodeDB(
                t=time,
                label=obj.label,
                pickle=node,
            )
        )
        gt_nodes.append(node)

    with write_lock if write_lock is not None else nullcontext():
        engine = sqla.create_engine(config.data_config.database_path)

        with Session(engine) as session:
            session.add_all(gt_db_rows)
            session.commit()

        engine.dispose()

        _link_gt(config, scale=scale, overwrite=False)

    # computing GT matching
    gt_matcher = SQLGTMatcher(config, write_lock=write_lock)
    total_score = gt_matcher()

    if len(gt_db_rows) > 0:
        mean_score = total_score / len(gt_db_rows)
    else:
        mean_score = 0.0

    LOG.info(f"time {time} total score: {total_score:0.4f}")
    LOG.info(f"time {time} mean score: {mean_score:0.4f}")


def _get_matched_nodes_df(database_path: str) -> pd.DataFrame:
    # TODO
    engine = sqla.create_engine(database_path)

    with Session(engine) as session:
        node_query = session.query(
            NodeDB.id,
            NodeDB.hier_parent_id,
            NodeDB.t_hier_id,
            NodeDB.area,
            NodeDB.frontier,
        ).where(NodeDB.t == 0)
        node_df = pd.read_sql(node_query.statement, session.bind, index_col="id")

        gt_query = session.query(GTLinkDB.source_id, GTLinkDB.target_id).where(
            GTLinkDB.selected
        )
        gt_df = pd.read_sql(gt_query.statement, session.bind, index="source_id")

    node_df = node_df.join(gt_df)

    frontiers = node_df["frontier"]
    node_df["parent_frontier"] = node_df["hier_parent_id"].map(
        lambda x: frontiers.get(x, -1.0)
    )
    node_df.loc[node_df["parent_frontier"] < 0, "parent_frontier"] = node_df[
        "frontier"
    ].max()

    return node_df


def match_to_ground_truth(
    config: MainConfig,
    gt_labels: ArrayLike,
    scale: Optional[ArrayLike] = None,
) -> pd.DataFrame:
    # TODO

    multiprocessing_apply(
        _match_ground_truth_frame(
            gt_labels=gt_labels,
            config=config,
            scale=scale,
        ),
        range(gt_labels.shape[0]),
        n_workers=config.segmentation_config.n_workers,
        desc="Matching hierarchy nodes with ground-truth",
    )

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


class SQLGTMatcher:
    def __init__(
        self,
        config: MainConfig,
        solver: Literal["CBC", "GUROBI", ""] = "",
        write_lock: Optional[fasteners.InterProcessLock] = None,
        eps=1e-3,
    ) -> None:
        # TODO

        self._data_config = config.data_config
        self._write_lock = write_lock
        self._eps = eps

        try:
            self._model = mip.Model(sense=mip.MAXIMIZE, solver_name=solver)
        except mip.exceptions.InterfacingError as e:
            LOG.warning(e)
            self._model = mip.Model(sense=mip.MAXIMIZE, solver_name="CBC")

    def _add_nodes(self) -> None:
        # TODO
        engine = sqla.create_engine(self._data_config.database_path)

        # t = 0 is hierarchies
        # t = 1 is ground-truth nodes
        with Session(engine) as session:
            query = session.query(NodeDB.id, NodeDB.t).where(NodeDB.t == 0)
            self._nodes_df = pd.read_sql(query.statement, session.bind, index_col="id")

        size = len(self._nodes_df)
        self._nodes = self._model.add_var_tensor(
            (size,), name="nodes", var_type=mip.BINARY
        )

        # hierarchy overlap constraints
        with Session(engine) as session:
            query = session.query(OverlapDB).join(
                NodeDB, NodeDB.id == OverlapDB.node_id
            )
            overlap_df = pd.read_sql(query.statement, session.bind)

        overlap_df["node_id"] = self._nodes_df.index.get_indexer(overlap_df["node_id"])
        overlap_df["ancestor_id"] = self._nodes_df.index.get_indexer(
            overlap_df["ancestor_id"]
        )

        for node_id, anc_id in zip(overlap_df["node_id"], overlap_df["ancestor_id"]):
            self._model.add_constr(self._nodes[node_id] + self._nodes[anc_id] <= 1)

    def _add_edges(self) -> None:
        # TODO

        if not hasattr(self, "_nodes"):
            raise ValueError("Nodes must be added before adding edges.")

        engine = sqla.create_engine(self._data_config.database_path)
        with Session(engine) as session:
            query = session.query(GTLinkDB).join(
                NodeDB, NodeDB.id == GTLinkDB.source_id
            )
            self._edges_df = pd.read_sql(query.statement, session.bind)

        self._edges_df["source_id"] = self._nodes_df.index.get_indexer(
            self._edges_df["source_id"]
        )
        self._edges_df.reset_index(drop=True, inplace=True)

        self._edges = self._model.add_var_tensor(
            (len(self._edges_df),),
            name="edges",
            var_type=mip.BINARY,
        )
        # small value to prefer not selecting edges than bad ones
        # setting objective function
        self._model.objective = mip.xsum(
            (self._edges_df["weight"].to_numpy() - self._eps) * self._edges
        )

        # source_id is time point T (hierarchies id)
        # target_id is time point T+1 (ground-truth)
        for source_id, group in self._edges_df.groupby("source_id", as_index=False):
            self._model.add_constr(
                self._nodes[source_id] == mip.xsum(self._edges[group.index.to_numpy()])
            )

        for _, group in self._edges_df.groupby("target_id", as_index=False):
            self._model.add_constr(mip.xsum(self._edges[group.index.to_numpy()]) <= 1)

    def add_solution(self) -> None:
        # TODO
        engine = sqla.create_engine(self._data_config.database_path)

        edges_records = []
        for idx, e_var in zip(self._edges_df.index, self._edges):
            if e_var.x > 0.5:
                edges_records.append(
                    {
                        "id": idx,
                        "selected": e_var.x > 0.5,
                    }
                )

        with self._write_lock if self._write_lock is not None else nullcontext():
            with Session(engine) as session:
                stmt = (
                    sqla.update(GTLinkDB)
                    .where(GTLinkDB.id == sqla.bindparam("id"))
                    .values(selected=sqla.bindparam("selected"))
                )
                session.connection().execute(
                    stmt,
                    edges_records,
                    execution_options={"synchronize_session": False},
                )
                session.commit()

    def __call__(self) -> float:
        # TODO
        self._add_nodes()
        self._add_edges()
        self._model.optimize()
        self.add_solution()

        n_selected_vars = sum(e_var.x > 0.5 for e_var in self._edges)

        return self._model.objective_value + n_selected_vars * self._eps
