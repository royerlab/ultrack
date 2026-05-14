import itertools
import logging
import math
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import sqlalchemy as sqla
from sqlalchemy.orm import Session

from ultrack.config.config import MainConfig
from ultrack.core.database import (
    NO_PARENT,
    GTLinkDB,
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


@dataclass(frozen=True)
class _BatchLayout:
    """Slice ranges used by a single windowed-solve batch.

    The solver builds variables for every time slice in ``[solver_start,
    solver_end]`` (the solver window). The database write covers
    ``[commit_start, commit_end]`` (the commit window). When a neighbouring
    batch has already committed its own solution, the boundary on that side is
    anchored: the solver shrinks to share exactly the anchored slice with the
    neighbour and the commit window does not extend past the inner range on
    that side.
    """

    solver_start: int
    solver_end: int
    commit_start: int
    commit_end: int
    left_anchored: bool
    right_anchored: bool


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
        self._layout: Optional[_BatchLayout] = None

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

        layout = self._compute_layout(index)
        self._layout = layout

        solver = MIPSolver(self._tracking_config)

        print(f"Solving ILP batch {index}")
        print(
            f"  solver window [{layout.solver_start}, {layout.solver_end}], "
            f"commit [{layout.commit_start}, {layout.commit_end}], "
            f"left_anchored={layout.left_anchored}, "
            f"right_anchored={layout.right_anchored}"
        )
        print("Constructing ILP ...")

        self._add_nodes(solver=solver, layout=layout)
        self._add_edges(solver=solver, layout=layout)

        solver.set_standard_constraints()

        self._add_overlap_constraints(solver=solver, layout=layout)
        self._add_boundary_constraints(solver=solver, layout=layout)

        self._solver = solver

    @property
    def solver(self) -> MIPSolver:
        if self._solver is None:
            raise ValueError("`construct_model` must be called first.")
        return self._solver

    def __call__(
        self,
        index: int,
        use_annotations: bool = False,
        use_ground_truth_match: bool = False,
    ) -> None:
        """Queries data from the given index, add it to the solver, run it and push to the database.

        Parameters
        ----------
        index : int
            Batch index.
        use_annotations : bool
            Fix ILP variables using annotations, by default False.
        use_ground_truth_match : bool
            Fix ILP variables using ground truth matching data, by default False.
        """
        self.construct_model(index)

        if use_annotations and use_ground_truth_match:
            raise ValueError(
                "Only one of `use_annotations` and `use_ground_truth_match` can be True."
            )

        if use_annotations:
            print("Fixing annotations ...")
            self.fix_annotations(index)
        elif use_ground_truth_match:
            print("Fixing ground truth matches ...")
            self.fix_ground_truth_matches(index)

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

        layout = (
            self._layout if self._layout is not None else self._compute_layout(index)
        )
        start_time, end_time = layout.solver_start, layout.solver_end

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

    def fix_ground_truth_matches(self, index: int) -> None:
        """
        Fix ILP variables using ground truth matching data.

        Parameters
        ----------
        index : int
            Batch index.
        """
        engine = sqla.create_engine(self._data_config.database_path)

        layout = (
            self._layout if self._layout is not None else self._compute_layout(index)
        )
        start_time, end_time = layout.solver_start, layout.solver_end

        with Session(engine) as session:
            # setting extra slack variables
            indices = [
                i
                for i, in session.query(GTLinkDB.source_id)  # equivalent to NodeDB.id
                .join(NodeDB, NodeDB.id == GTLinkDB.source_id)
                .where(NodeDB.t.between(start_time, end_time), GTLinkDB.selected)
            ]
            self.solver.enforce_nodes_solution_value(indices, "node", True)

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

        Kept for backward compatibility; ``_compute_layout`` is the new
        canonical helper that also accounts for whether neighbouring batches
        have already committed.

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

    def _is_committed_at(self, t: int) -> bool:
        """Whether some node at time ``t`` is already ``selected`` in the DB.

        Used to detect that a neighbouring batch has already committed its
        solution at the boundary of the current batch.
        """
        if t < 0 or t > self._max_t:
            return False
        engine = sqla.create_engine(self._data_config.database_path)
        with Session(engine) as session:
            return (
                session.query(NodeDB.id).where(NodeDB.t == t, NodeDB.selected).first()
                is not None
            )

    def _compute_layout(self, index: int) -> _BatchLayout:
        """Decide solver and commit windows for ``index`` from DB state.

        A batch is *anchored* on a side when its neighbour batch on that side
        has already written a solution there. Anchored batches use a tight
        window so the neighbour's committed boundary is reused verbatim;
        non-anchored sides use the configured overlap on the solver window
        and commit one slice past the inner range so the next-solving
        neighbour finds an anchor.
        """
        overlap = self._tracking_config.overlap_size
        inner_start = index * self._window_size
        inner_end = min((index + 1) * self._window_size - 1, self._max_t)

        left_anchored = inner_start > 0 and self._is_committed_at(inner_start - 1)
        right_anchored = inner_end < self._max_t and self._is_committed_at(
            inner_end + 1
        )

        if left_anchored:
            solver_start = inner_start
            commit_start = inner_start
        else:
            solver_start = max(inner_start - overlap, 0)
            commit_start = max(inner_start - 1, 0)

        if right_anchored:
            solver_end = inner_end
            commit_end = inner_end
        else:
            solver_end = min(inner_end + overlap, self._max_t)
            commit_end = min(inner_end + 1, self._max_t)

        return _BatchLayout(
            solver_start=solver_start,
            solver_end=solver_end,
            commit_start=commit_start,
            commit_end=commit_end,
            left_anchored=left_anchored,
            right_anchored=right_anchored,
        )

    def _add_nodes(self, solver: BaseSolver, layout: _BatchLayout) -> None:
        """Query nodes inside ``layout.solver_*`` and add them to the solver.

        Appearance and disappearance are kept free at the solver window
        boundaries only when that side is *not* anchored to a neighbouring
        batch's committed selection; otherwise the solver pays the regular
        penalty so it cannot spawn or terminate tracks at an interior seam.
        """
        start_time = layout.solver_start
        end_time = layout.solver_end

        engine = sqla.create_engine(self._data_config.database_path)
        border_distance = self._tracking_config.image_border_size
        shape = self._data_config.metadata["shape"]
        data_dim = len(shape)

        if border_distance is None or np.all(border_distance == 0):
            with Session(engine) as session:
                query = session.query(
                    NodeDB.id,
                    NodeDB.t,
                    NodeDB.node_prob,
                ).where(NodeDB.t.between(start_time, end_time))
                df = pd.read_sql(query.statement, session.bind)
            is_border = False
        else:
            # Handle border distance
            # Ensure border_distance and shape have the same dimension
            if len(border_distance) < 3:
                border_distance = (0,) * (3 - len(border_distance)) + border_distance

            if data_dim == 3:
                # TYX -> TZYX
                shape = (shape[-3], 1, shape[-2], shape[-1])

            with Session(engine) as session:
                query = session.query(
                    NodeDB.id,
                    NodeDB.t,
                    NodeDB.z,
                    NodeDB.y,
                    NodeDB.x,
                    NodeDB.node_prob,
                ).where(NodeDB.t.between(start_time, end_time))
                df = pd.read_sql(query.statement, session.bind)

            is_border = _check_inside_border(df, border_distance, shape)

        n_invalid_prob = (df["node_prob"] < 0).sum()

        if n_invalid_prob == df.shape[0]:
            nodes_prob = None
        elif n_invalid_prob == 0:
            nodes_prob = df["node_prob"]
        else:
            raise ValueError(
                "None or all nodes' probabilities must be provided found "
                f"Found {df.shape[0] - n_invalid_prob} / {df.shape[0]} valid probs."
            )

        LOG.info(f"nodes with t between {start_time} and {end_time}")

        solver.add_nodes(
            df["id"],
            df["t"] == start_time,
            df["t"] == end_time,
            is_border=is_border,
            nodes_prob=nodes_prob,
            free_appear=not layout.left_anchored,
            free_disappear=not layout.right_anchored,
        )

    def _add_edges(self, solver: BaseSolver, layout: _BatchLayout) -> None:
        """Query edges inside ``layout.solver_*`` and add them to the solver."""
        start_time = layout.solver_start
        end_time = layout.solver_end

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
            f"edges with source nodes with t between {start_time} and {end_time - 1}"
        )

        solver.add_edges(df["source_id"], df["target_id"], df["weight"])

    def _add_overlap_constraints(
        self, solver: BaseSolver, layout: _BatchLayout
    ) -> None:
        """Adds overlapping segmentation constraints inside the solver window."""
        start_time = layout.solver_start
        end_time = layout.solver_end

        engine = sqla.create_engine(self._data_config.database_path)
        with Session(engine) as session:
            query = (
                session.query(OverlapDB)
                .join(NodeDB, NodeDB.id == OverlapDB.node_id)
                .where(NodeDB.t.between(start_time, end_time))
            )
            df = pd.read_sql(query.statement, session.bind)

        solver.add_overlap_constraints(df["node_id"], df["ancestor_id"])

    def _add_boundary_constraints(
        self, solver: BaseSolver, layout: _BatchLayout
    ) -> None:
        """Force this batch's solver to keep the neighbours' boundary picks.

        At an anchored side the solver window's outermost slice is shared with
        the neighbouring batch's already-committed selection, so every
        ``selected=True`` node at that slice is forced into this batch's
        solution. The non-anchored sides have no neighbouring commit to read.
        """
        engine = sqla.create_engine(self._data_config.database_path)
        with Session(engine) as session:
            query = session.query(NodeDB.id).where(NodeDB.selected)

            start_nodes = [n for n, in query.where(NodeDB.t == layout.solver_start)]
            end_nodes = [n for n, in query.where(NodeDB.t == layout.solver_end)]

        LOG.info(
            f"# {len(start_nodes)} boundary constraints found at t = {layout.solver_start}"
        )
        LOG.info(
            f"# {len(end_nodes)} boundary constraints found at t = {layout.solver_end}"
        )

        solver.enforce_nodes_solution_value(start_nodes, variable="node", value=True)
        solver.enforce_nodes_solution_value(end_nodes, variable="node", value=True)

    def add_solution(self, index: int = 0) -> None:
        """Adds selected nodes to solution in database.

        Writes this batch's solution over ``layout.commit_start..commit_end``.
        ``parent_id`` is updated everywhere in the commit range except at the
        leftmost slice when it coincides with the solver's leftmost slice --
        there the solver has no incoming edge and its ``parent_id`` is
        ``NO_PARENT``, which would clobber the previous batch's lineage.

        Parameters
        ----------
        index : int
            Batch index, by default 0, which works for single batch tracking.
        """
        solution = self.solver.solution()
        solution["node_id"] = solution.index

        layout = (
            self._layout if self._layout is not None else self._compute_layout(index)
        )
        commit_start = layout.commit_start
        commit_end = layout.commit_end
        skip_parent_at_start = layout.commit_start == layout.solver_start

        records = solution[["node_id", "parent_id"]].to_dict("records")
        ids_only = solution[["node_id"]].to_dict("records")

        engine = sqla.create_engine(self._data_config.database_path)
        with Session(engine) as session:
            # Inner range: write selected + parent_id for every batch node.
            inner_lower = commit_start + 1 if skip_parent_at_start else commit_start
            if inner_lower <= commit_end:
                inner_stmt = (
                    sqla.update(NodeDB)
                    .where(
                        NodeDB.t.between(inner_lower, commit_end),
                        NodeDB.id == sqla.bindparam("node_id"),
                    )
                    .values(parent_id=sqla.bindparam("parent_id"), selected=True)
                )
                session.connection().execute(
                    inner_stmt,
                    records,
                    execution_options={"synchronize_session": False},
                )

            # Left boundary slice (only when the solver had no incoming edges
            # here): mark selected without touching parent_id so we keep the
            # previous batch's lineage pointing back into its own inner range.
            if skip_parent_at_start:
                boundary_stmt = (
                    sqla.update(NodeDB)
                    .where(
                        NodeDB.t == commit_start,
                        NodeDB.id == sqla.bindparam("node_id"),
                    )
                    .values(selected=True)
                )
                session.connection().execute(
                    boundary_stmt,
                    ids_only,
                    execution_options={"synchronize_session": False},
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


def _check_inside_border(
    df: pd.DataFrame,
    border_distance: Tuple[int, int, int],
    shape: Tuple[int, int, int, int],
) -> np.ndarray:
    """
    Check if nodes are inside the border.

    Parameters
    ----------
    df : pd.DataFrame
        Nodes dataframe.
    border_distance : Tuple[int, int, int]
        Border distance in pixels (Z, Y, X).
    shape : Tuple[int, int, int, int]
        Image shape (T, Z, Y, X).

    Returns
    -------
    np.ndarray
        Boolean array indicating if nodes are inside the border.
    """
    inside_border = np.logical_and.reduce(
        [
            (border_distance[i] <= df[c]) & (df[c] <= shape[i] - border_distance[i])
            for i, c in zip([-3, -2, -1], ["z", "y", "x"])
        ]
    )
    is_border = ~inside_border
    # Log the coordinates of nodes inside the border if logging level is debug
    if LOG.isEnabledFor(logging.DEBUG):
        for row in df[is_border].itertuples(index=False):
            LOG.debug(
                "Node with coords (z,y,x) at: (%s, %s, %s) is inside border.",
                row.z,
                row.y,
                row.x,
            )
    return is_border
