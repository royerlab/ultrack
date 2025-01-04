import logging
from contextlib import nullcontext
from typing import Optional

import fasteners
import mip
import mip.exceptions
import pandas as pd
import sqlalchemy as sqla
from sqlalchemy.orm import Session

from ultrack.config.config import MainConfig
from ultrack.core.database import GTLinkDB, NodeDB, OverlapDB

LOG = logging.getLogger(__name__)


class SQLGTMatcher:
    def __init__(
        self,
        config: MainConfig,
        write_lock: Optional[fasteners.InterProcessLock] = None,
        eps: float = 1e-3,
    ) -> None:
        """
        Ground-truth matching solver from SQL database content.

        Parameters
        ----------
        config : MainConfig
            Ultrack's main configuration parameters.
        write_lock : Optional[fasteners.InterProcessLock], optional
            Lock object for SQLite multiprocessing.
        eps : float, optional
            Weight epsilon to prefer not selecting bad matches than bad ones.
        """
        self._data_config = config.data_config
        self._write_lock = write_lock
        self._connect_args = {"timeout": 45} if self._write_lock is not None else {}
        self._eps = eps

        try:
            self._model = mip.Model(
                sense=mip.MAXIMIZE, solver_name=config.tracking_config.solver_name
            )
        except mip.exceptions.InterfacingError as e:
            LOG.warning(e)
            self._model = mip.Model(sense=mip.MAXIMIZE, solver_name="CBC")

    def _add_nodes(self, time: int) -> None:
        """
        Add nodes to the ILP model.

        Parameters
        ----------
        time : int
            Time point to query and match.
        """
        engine = sqla.create_engine(
            self._data_config.database_path, connect_args=self._connect_args
        )

        with Session(engine) as session:
            query = session.query(NodeDB.id, NodeDB.t).where(NodeDB.t == time)
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

    def _add_edges(self, time: int) -> None:
        """
        Add edges to the ILP model.

        Parameters
        ----------
        time : int
            Time point to query and match.
        """
        if not hasattr(self, "_nodes"):
            raise ValueError("Nodes must be added before adding edges.")

        engine = sqla.create_engine(
            self._data_config.database_path, connect_args=self._connect_args
        )

        with Session(engine) as session:

            query = (
                session.query(GTLinkDB)
                .join(NodeDB, NodeDB.id == GTLinkDB.source_id)
                .where(NodeDB.t == time)
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
        """
        Add the solution to the database.
        """
        engine = sqla.create_engine(
            self._data_config.database_path, connect_args=self._connect_args
        )

        edges_records = []
        for idx, e_var in zip(self._edges_df["id"], self._edges):
            if e_var.x > 0.5:
                edges_records.append(
                    {
                        "link_id": idx,
                        "selected": e_var.x > 0.5,
                    }
                )

        LOG.info(f"Selected {len(edges_records)} edges to ground-truth")

        with self._write_lock if self._write_lock is not None else nullcontext():
            with Session(engine) as session:
                stmt = (
                    sqla.update(GTLinkDB)
                    .where(GTLinkDB.id == sqla.bindparam("link_id"))
                    .values(selected=sqla.bindparam("selected"))
                )
                session.connection().execute(
                    stmt,
                    edges_records,
                    execution_options={"synchronize_session": False},
                )
                session.commit()

    def __call__(self, time: int) -> float:
        """
        Build the ground-truth matching ILP and solve it.

        Parameters
        ----------
        time : int
            Time point to query and match.

        Returns
        -------
        float
            Objective value.
        """
        LOG.info(f"Computing GT matching for time {time}")

        self._add_nodes(time)
        self._add_edges(time)
        self._model.optimize()
        self.add_solution()

        n_selected_vars = sum(e_var.x > 0.5 for e_var in self._edges)

        return self._model.objective_value + n_selected_vars * self._eps
