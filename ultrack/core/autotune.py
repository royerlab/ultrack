import logging
from typing import Literal, Optional, Tuple

import mip
import mip.exceptions
import numpy as np
import pandas as pd
import sqlalchemy as sqla
from numpy.typing import ArrayLike
from skimage.measure import regionprops
from sqlalchemy.orm import Session
from toolz import curry

from ultrack.config.config import MainConfig
from ultrack.core.database import LinkDB, NodeDB, OverlapDB, clear_all_data
from ultrack.core.interactive import add_new_node
from ultrack.core.linking.processing import link
from ultrack.core.segmentation.processing import segment
from ultrack.tracks.stats import estimate_drift
from ultrack.utils.multiprocessing import multiprocessing_apply

LOG = logging.getLogger(__name__)


class SQLGTMatching:
    def __init__(
        self,
        config: MainConfig,
        solver: Literal["CBC", "GUROBI", ""] = "",
    ) -> None:
        # TODO

        self._data_config = config.data_config

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
            query = session.query(LinkDB).join(NodeDB, NodeDB.id == LinkDB.source_id)
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
        # setting objective function
        self._model.objective = mip.xsum(
            self._edges_df["weight"].to_numpy() * self._edges
        )

        # source_id is time point T (hierarchies id)
        # target_id is time point T+1 (ground-truth)
        for source_id, group in self._edges_df.groupby("source_id", as_index=False):
            self._model.add_constr(
                self._nodes[source_id] == mip.xsum(self._edges[group.index.to_numpy()])
            )

        for _, group in self._edges_df.groupby("target_id", as_index=False):
            self._model.add_constr(mip.xsum(self._edges[group.index.to_numpy()]) <= 1)

    def __call__(self) -> Tuple[float, pd.DataFrame]:
        # TODO
        self._add_nodes()
        self._add_edges()
        self._model.optimize()

        data = []

        for i, e_var in enumerate(self._edges):
            if e_var.x > 0.5:
                data.append(
                    {
                        "id": self._nodes_df.index.get_indexer(
                            self._edges_df.iloc[i]["source_id"]
                        ),
                        "gt_id": self._edges_df.iloc[i]["target_id"],
                    }
                )

        score = self._model.objective_value
        matching_df = pd.DataFrame(data)

        return score, matching_df


@curry
def _tune_time_point(
    t: int,
    foreground: ArrayLike,
    contours: ArrayLike,
    gt_labels: ArrayLike,
    config: MainConfig,
    scale: Optional[ArrayLike],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # TODO

    config = config.copy(deep=True)
    config.data_config.in_memory_db_id = t

    clear_all_data(config.data_config.database_path)

    gt_labels = np.asarray(gt_labels[t])
    gt_rows = []

    props = regionprops(gt_labels)

    if len(props) == 0:
        LOG.warning(f"No objects found in time point {t}")

    foreground = np.asarray(foreground[t])
    contours = np.asarray(contours[t])

    # adding hierarchy nodes
    segment(
        foreground=foreground[None, ...],
        contours=contours[None, ...],
        config=config,
        overwrite=False,
    )

    # adding ground-truth nodes
    for obj in props:
        add_new_node(
            config=config,
            time=1,
            mask=obj.image,
            bbox=obj.bbox,
            index=obj.label,  # _generate_id(obj.label, 1, 10_000_000),
            include_overlaps=False,
        )
        row = {c: v for c, v in zip("xyz", obj.centroid[::-1])}
        row["track_id"] = obj.label
        gt_rows.append(row)

    gt_df = pd.DataFrame.from_records(gt_rows)
    gt_df["t"] = t

    # computing GT matching
    link(config, scale=scale, overwrite=False)

    matching = SQLGTMatching(config)
    total_score, solution_df = matching()

    if len(gt_df) > 0:
        mean_score = total_score / len(gt_df)
    else:
        mean_score = 0.0

    print(f"Total score: {total_score:0.4f}")
    print(f"Mean score: {mean_score:0.4f}")

    engine = sqla.create_engine(config.data_config.database_path)

    with Session(engine) as session:
        query = session.query(
            NodeDB.id,
            NodeDB.hier_parent_id,
            NodeDB.t_hier_id,
            NodeDB.area,
            NodeDB.frontier,
        ).where(NodeDB.t == 0)

        df = pd.read_sql(query.statement, session.bind, index_col="id")

    df = df.join(solution_df)

    frontiers = df["frontier"]

    df["parent_frontier"] = df["hier_parent_id"].map(lambda x: frontiers.get(x, -1.0))
    df.loc[df["parent_frontier"] < 0, "parent_frontier"] = df["frontier"].max()

    # selecting only nodes in solution
    # must be after parent_frontier computation
    # matched_df = df[df["solution"] > 0.5]

    # config.segmentation_config.min_frontier = matched_df["parent_frontier"].min()
    # config.segmentation_config.min_area = matched_df["area"].min()
    # config.segmentation_config.max_area = matched_df["area"].max()

    # config.data_config.in_memory_db_id = prev_in_memory_db_id

    return df, gt_df


def auto_tune_config(
    foreground: ArrayLike,
    contours: ArrayLike,
    ground_truth_labels: ArrayLike,
    config: Optional[MainConfig] = None,
    scale: Optional[ArrayLike] = None,
) -> Tuple[MainConfig, pd.DataFrame]:

    if config is None:
        config = MainConfig()
    else:
        config = config.copy(deep=True)

    prev_db = config.data_config.database
    config.data_config.database = "memory"

    tuning_tup = multiprocessing_apply(
        _tune_time_point(
            foreground=foreground,
            contours=contours,
            gt_labels=ground_truth_labels,
            config=config,
            scale=scale,
        ),
        range(foreground.shape[0]),
        n_workers=config.segmentation_config.n_workers,
        desc="Auto-tuning individual time points",
    )
    tuning_tup = tuple(zip(*tuning_tup))
    df = pd.concat(tuning_tup[0], ignore_index=True)
    gt_df = pd.concat(tuning_tup[1], ignore_index=True)

    if scale is not None:
        cols = ["z", "y", "x"][-len(scale) :]
        gt_df[cols] *= scale

    if "z" not in gt_df.columns:
        gt_df["z"] = 0.0

    if len(gt_df) > 0:
        max_distance = estimate_drift(gt_df)
        if not np.isnan(max_distance) or max_distance > 0:
            config.linking_config.max_distance = max_distance + 1.0

    if "solution" in df.columns:
        matched_df = df[df["solution"] > 0.5]
        config.segmentation_config.min_area = matched_df["area"].min() * 0.95

        config.segmentation_config.max_area = matched_df["area"].max() * 1.025

        config.segmentation_config.min_frontier = max(
            matched_df["parent_frontier"].min() - 0.025, 0.0
        )
    else:
        LOG.warning("No nodes were matched. Keeping previous configuration.")

    config.data_config.database = prev_db

    return config, df
