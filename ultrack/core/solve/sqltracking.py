import itertools
import logging
import math
from typing import Optional, Tuple

import pandas as pd
import sqlalchemy as sqla
from mip.exceptions import InterfacingError
from sqlalchemy.orm import Session

from ultrack.config.config import MainConfig
from ultrack.core.database import (
    NO_PARENT,
    LinkDB,
    NodeDB,
    OverlapDB,
    VarAnnotation,
    maximum_time_from_database,
)
from ultrack.core.solve.solver import MIPSolver
from ultrack.core.solve.solver.base_solver import BaseSolver

logging.basicConfig()
logging.getLogger("sqlachemy.engine").setLevel(logging.INFO)

LOG = logging.getLogger(__name__)

_KEY_TO_SOLVER_NAME = {
    "CBC": "Coin-OR Branch and Cut",
    "GRB": "Gurobi",
}


class SQLTracking:
    def __init__(
        self,
        config: MainConfig,
    ) -> None:
        """
        Helper class to query data from SQL database and dispatch do solver.

        Parameters
        ----------
        config : MainConfig
            Configuration parameters.
        """
        LOG.info(f"SQLTracking with TrackingConfig:\n{config.tracking_config}")

        self._tracking_config = config.tracking_config
        self._data_config = config.data_config
        self._solver: Optional[MIPSolver] = None

        self._max_t = maximum_time_from_database(self._data_config)
        if self._tracking_config.window_size is None:
            LOG.info(f"Window size not set, configured to {self._max_t + 1}.")
            self._window_size = self._max_t + 1
        else:
            self._window_size = self._tracking_config.window_size

        self.num_batches = math.ceil(self._max_t / self._window_size)

    def construct_model(self, index: int = 0) -> None:
        """
        Constructs ILP model for a given batch index.
        Adding nodes, edges and slack variables, and biological, boundary and overlap constraints.

        Parameters
        ----------
        index : int
            Batch index, by default 0, which works for single batch tracking.
        """
        if index < 0 or index >= self.num_batches:
            raise ValueError(
                f"Invalid index {index}, expected between [0, {self.num_batches})."
            )

        try:
            solver = MIPSolver(self._tracking_config)
        except InterfacingError as e:
            LOG.warning(e)
            solver = MIPSolver(self._tracking_config, "CBC")

        print(f"Using {_KEY_TO_SOLVER_NAME[solver._model.solver_name]} solver")
        print(f"Solving ILP batch {index}")
        print("Constructing ILP ...")

        self._add_nodes(solver=solver, index=index)
        self._add_edges(solver=solver, index=index)

        solver.set_standard_constraints()

        self._add_overlap_constraints(solver=solver, index=index)
        self._add_boundary_constraints(solver=solver, index=index)

        self._solver = solver

    @property
    def solver(self) -> MIPSolver:
        if self._solver is None:
            raise ValueError("`construct_model` must be called first.")
        return self._solver

    def __call__(self, index: int, use_annotations: bool = False) -> None:
        """Queries data from the given index, add it to the solver, run it and push to the database.

        Parameters
        ----------
        index : int
            Batch index.
        use_annotations : bool
            Fix ILP variables using annotations, by default False.
        """
        self.construct_model(index)

        if use_annotations:
            print("Fixing annotations ...")
            self.fix_annotations(index)

        print("Solving ILP ...")
        self.solve()

        print("Saving solution ...")
        self.add_solution(index)

        print("Done!")

    def fix_annotations(self, index: int) -> None:
        """
        Fix ILP variables using annotations.

        Parameters
        ----------
        index : int
            Batch index.
        """

        engine = sqla.create_engine(self._data_config.database_path)

        start_time, end_time = self._window_limits(index, True)

        with Session(engine) as session:
            # setting extra slack variables
            for mode, value in itertools.product(
                ["appear", "disappear", "division", "node"],
                [True, False],
            ):
                column = getattr(NodeDB, f"{mode}_annot")
                enum_value = VarAnnotation.REAL if value else VarAnnotation.FAKE
                indices = [
                    i
                    for i, in session.query(NodeDB.id).where(
                        NodeDB.t.between(start_time, end_time), column == enum_value
                    )
                ]
                self.solver.enforce_nodes_solution_value(indices, mode, value)

            # setting links
            for value in (True, False):
                enum_value = VarAnnotation.REAL if value else VarAnnotation.FAKE
                query = (
                    session.query(LinkDB.source_id, LinkDB.target_id)
                    .join(NodeDB, NodeDB.id == LinkDB.source_id)
                    .where(
                        NodeDB.t.between(start_time, end_time),
                        LinkDB.annotation == enum_value,
                    )
                )
                df = pd.read_sql(query.statement, session.bind)
                self.solver.enforce_edges_solution_value(
                    df["source_id"],
                    df["target_id"],
                    value,
                )

    def solve(self) -> None:
        """Tracks by solving optimization model."""
        self.solver.optimize()

    def set_number_of_segments(self, time: int, number_of_segments: int) -> None:
        """Sets as a constraint the number of segments for a given time point.

        Parameters
        ----------
        time : int
            Time point.
        number_of_segments : int
            Number of segments.

        Examples
        --------
        >>> # creating tracking problem
        >>> segment(foreground, edges, config)
        >>> link(config)
        >>>
        >>> # low level API tracking
        >>> tracker = SQLTracking(config)
        >>> tracker.construct_model()
        >>> tracker.set_number_of_segments(0, 2)
        >>> tracker.solve()
        >>> tracker.add_solution()
        """
        engine = sqla.create_engine(self._data_config.database_path)
        with Session(engine) as session:
            indices = [i for i, in session.query(NodeDB.id).where(NodeDB.t == time)]

        self.solver.set_nodes_sum(indices, number_of_segments)

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

    def _add_nodes(self, solver: BaseSolver, index: int) -> None:
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

        start_time = max(start_time, 0)
        end_time = min(end_time, self._max_t)

        LOG.info(f"Batch {index}, nodes with t between {start_time} and {end_time}")

        solver.add_nodes(
            df["id"],
            df["t"] == start_time,
            df["t"] == end_time,
        )

    def _add_edges(self, solver: BaseSolver, index: int) -> None:
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
                .where(NodeDB.t.between(start_time, end_time - 1))
                # subtracting one because we're using source_id as reference
            )
            df = pd.read_sql(query.statement, session.bind)

        LOG.info(
            f"Batch {index}, edges with source nodes with t between {start_time} and {end_time - 1}"
        )

        solver.add_edges(df["source_id"], df["target_id"], df["weight"])

    def _add_overlap_constraints(self, solver: BaseSolver, index: int) -> None:
        """Adds overlaping segmentation constrainsts

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

        solver.add_overlap_constraints(df["node_id"], df["ancestor_id"])

    def _add_boundary_constraints(self, solver: BaseSolver, index: int) -> None:
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

        solver.enforce_nodes_solution_value(start_nodes, variable="node", value=True)
        solver.enforce_nodes_solution_value(end_nodes, variable="node", value=True)

    def add_solution(self, index: int = 0) -> None:
        """Adds selected nodes to solution in database.

        Parameters
        ----------
        index : int
            Batch index, by default 0, which works for single batch tracking.
        """
        solution = self.solver.solution()

        solution["node_id"] = solution.index

        start_time, end_time = self._window_limits(index, False)

        engine = sqla.create_engine(self._data_config.database_path)
        with Session(engine) as session:
            general_stmt = (
                sqla.update(NodeDB)
                .where(
                    NodeDB.t.between(start_time, end_time),
                    NodeDB.id == sqla.bindparam("node_id"),
                )
                .values(parent_id=sqla.bindparam("parent_id"), selected=True)
            )
            session.execute(
                general_stmt,
                solution[["node_id", "parent_id"]].to_dict("records"),
                execution_options={"synchronize_session": False},
            )

            # condition isn't necessary but avoids a useless operation
            if start_time > 0:
                # insert nodes from start time - 1 without their parent
                start_stmt = (
                    sqla.update(NodeDB)
                    .where(
                        NodeDB.t == start_time - 1,
                        NodeDB.id == sqla.bindparam("node_id"),
                    )
                    .values(selected=True)
                )
                session.execute(
                    start_stmt,
                    solution[["node_id"]].to_dict("records"),
                    execution_options={"syncronize_session": False},
                )

            session.commit()

    def reset_solution(self) -> None:
        """Resets every node parent_id and selected variables."""
        self.clear_solution_from_database(self._data_config.database_path)

    @staticmethod
    def clear_solution_from_database(database_path: str) -> None:
        LOG.info("Clearing nodes database solutions.")
        engine = sqla.create_engine(database_path)
        with Session(engine) as session:
            statement = sqla.update(NodeDB).values(parent_id=NO_PARENT, selected=False)
            session.execute(statement)
            session.commit()
