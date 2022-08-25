import logging
import math
from typing import Tuple

import pandas as pd
import sqlalchemy as sqla
from sqlalchemy.orm import Session

from ultrack.config.dataconfig import DataConfig
from ultrack.config.trackingconfig import TrackingConfig
from ultrack.core.database import NO_PARENT, LinkDB, NodeDB, OverlapDB, maximum_time
from ultrack.core.tracking.gurobi_solver import GurobiSolver

logging.basicConfig()
logging.getLogger("sqlachemy.engine").setLevel(logging.INFO)

LOG = logging.getLogger(__name__)


class SQLTracking:
    def __init__(
        self, tracking_config: TrackingConfig, data_config: DataConfig
    ) -> None:
        """
        Helper class to query data from SQL database and dispatch do solver.

        Parameters
        ----------
        tracking_config : TrackingConfig
            Tracking configuration parameters.
        data_config : DataConfig
            Data configuration parameters.
        """
        LOG.info(f"SQLTracking with TrackingConfig:\n{tracking_config}")

        self._tracking_config = tracking_config
        self._data_config = data_config
        self._solver = GurobiSolver(self._tracking_config)

        self._max_t = maximum_time(self._data_config)
        if self._tracking_config.window_size is None:
            LOG.info(f"Window size not set, configured to {self._max_t + 1}.")
            self._window_size = self._max_t + 1
        else:
            self._window_size = self._tracking_config.window_size

        self.num_batches = math.ceil(self._max_t / self._window_size)

    def __call__(self, index: int) -> None:
        """Queries data from the given index, add it to the solver, run it and push to the database.

        Parameters
        ----------
        index : int
            Batch index.
        """
        if index < 0 or index >= self.num_batches:
            raise ValueError(
                f"Invalid index {index}, expected between [0, {self.num_batches})."
            )

        LOG.info(f"Tracking batch {index}")
        self._solver.reset()
        self._add_nodes(index=index)
        self._add_edges(index=index)

        self._solver.set_standard_constraints()
        self._add_overlap_constraints(index=index)
        self._add_boundary_constraints(index=index)

        self._solver.optimize()

        self._update_solution(index, self._solver.solution())

    def _window_limits(self, index: int, with_overlap: bool) -> Tuple[int, int]:
        """Computes time window of a given index, with or without overlap.

        Parameters
        ----------
        index : int
            Batch index.
        with_overlap : bool
            Flag indicating if `overlap_size` should be taken into account or not.

        Returns
        -------
        Tuple[int, int]
            Lower and upper window boundary.
        """
        start_time = (
            index * self._window_size
            - with_overlap * self._tracking_config.overlap_size
        )
        end_time = (
            index + 1
        ) * self._window_size + with_overlap * self._tracking_config.overlap_size
        return start_time, end_time - 1

    def _add_nodes(self, index: int) -> None:
        """Query nodes from a given batch index and add them to solver.

        Parameters
        ----------
        index : int
            Batch index.
        """
        start_time, end_time = self._window_limits(index, True)

        engine = sqla.create_engine(self._data_config.database_path)
        with Session(engine) as session:
            query = session.query(NodeDB.id, NodeDB.t).where(
                NodeDB.t.between(start_time, end_time)
            )
            df = pd.read_sql(query.statement, session.bind)

        self._solver.add_nodes(
            df["id"],
            df["t"] == max(start_time, 0),
            df["t"] == min(end_time, self._max_t),
        )

    def _add_edges(self, index: int) -> None:
        """Query edges from a given batch index and add them to solver.

        Parameters
        ----------
        index : int
            Batch index.
        """
        start_time, end_time = self._window_limits(index, True)

        engine = sqla.create_engine(self._data_config.database_path)
        with Session(engine) as session:
            query = (
                session.query(LinkDB)
                .join(NodeDB, NodeDB.id == LinkDB.source_id)
                .where(NodeDB.t.between(start_time, end_time))
                # subtracting one because we're using source_id as reference
            )
            df = pd.read_sql(query.statement, session.bind)

        self._solver.add_edges(df["source_id"], df["target_id"], df["iou"])

    def _add_overlap_constraints(self, index: int) -> None:
        """Adds overlap and standard biological contraints.

        Parameters
        ----------
        index : int
            Batch index.
        """
        start_time, end_time = self._window_limits(index, True)

        engine = sqla.create_engine(self._data_config.database_path)
        with Session(engine) as session:
            query = (
                session.query(OverlapDB)
                .join(NodeDB, NodeDB.id == OverlapDB.node_id)
                .where(NodeDB.t.between(start_time, end_time))
            )
            df = pd.read_sql(query.statement, session.bind)

        self._solver.add_overlap_constraints(df["node_id"], df["ancestor_id"])

    def _add_boundary_constraints(self, index: int) -> None:
        """
        Enforce to solution nodes from the boundary (in time) already selected from adjacent batches.

        Parameters
        ----------
        index : int
            Batch index.
        """
        start_time, end_time = self._window_limits(index, True)

        engine = sqla.create_engine(self._data_config.database_path)
        with Session(engine) as session:
            query = session.query(NodeDB.id).where(NodeDB.selected)

            start_nodes = [n for n, in query.where(NodeDB.t == start_time)]
            end_nodes = [n for n, in query.where(NodeDB.t == end_time)]

        LOG.info(
            f"# {len(start_nodes)} boundary constraints found at at t = {start_time}"
        )
        LOG.info(f"# {len(end_nodes)} boundary constraints found at at t = {end_time}")

        self._solver.enforce_node_to_solution(start_nodes)
        self._solver.enforce_node_to_solution(end_nodes)

    def _update_solution(self, index: int, solution: pd.DataFrame) -> None:
        """Updates solution in database.

        Parameters
        ----------
        index : int
            Batch index.
        solution : pd.DataFrame
            Dataframe indexed by nodes' id containing parent_id column.
        """
        solution["node_id"] = solution.index

        start_time, end_time = self._window_limits(index, False)
        stmt = (
            sqla.update(NodeDB)
            .where(
                NodeDB.t.between(start_time, end_time),
                NodeDB.id == sqla.bindparam("node_id"),
            )
            .values(parent_id=sqla.bindparam("parent_id"), selected=True)
        )

        engine = sqla.create_engine(self._data_config.database_path)
        with Session(engine) as session:
            session.execute(
                stmt,
                solution[["node_id", "parent_id"]].to_dict("records"),
                execution_options={"synchronize_session": False},
            )
            session.commit()

    def reset_solution(self) -> None:
        """Resets every node parent_id and selected variables."""
        LOG.info("Resetting database solutions.")
        engine = sqla.create_engine(self._data_config.database_path)
        with Session(engine) as session:
            statement = sqla.update(NodeDB).values(parent_id=NO_PARENT, selected=False)
            session.execute(statement)
            session.commit()
