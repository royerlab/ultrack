import logging

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import sparse
from skimage.util._map_array import ArrayMap

from ultrack.config.config import TrackingConfig
from ultrack.core.solve.solver.base_solver import BaseSolver
from ultrack.core.solve.solver.heuristic._numba_heuristic_solver import (
    NumbaHeuristicSolver,
)
from ultrack.utils.array import assert_same_length
from ultrack.utils.constants import NO_PARENT

LOG = logging.getLogger(__name__)


class HeuristicSolver(BaseSolver):
    def __init__(
        self,
        config: TrackingConfig,
    ) -> None:
        """
        Heuristic solver for cell-tracking ILP.
        This algorithm computes an approximate solution using Kruskal MST algorithm
        while respecting the biological constraints. Hence, this solution isn't a MST.
        Then, it performs multiple random ascent by performing local searches on random nodes.

        Parameters
        ----------
        config : TrackingConfig
            Tracking configuration parameters.
        """

        self._config = config
        self._rng = np.random.default_rng(42)
        self._objective = 0.0

        # w_ap, w_dap, w_div
        self._add_in_map = np.full((2, 3, 3), fill_value=-1e6, dtype=np.float32)
        self._add_in_map[0, 0] = (0, 1, 0)
        self._add_in_map[0, 1:] = (-1, 0, 0)

        self._sub_in_map = -np.roll(self._add_in_map, shift=1, axis=0)

        self._add_out_map = np.full((2, 3, 3), fill_value=-1e6, dtype=np.float32)
        self._add_out_map[0, 0] = (1, 0, 0)
        self._add_out_map[1, 0] = (0, -1, 0)
        self._add_out_map[:, 1] = (0, 0, 1)

        self._sub_out_map = -np.roll(self._add_out_map, shift=1, axis=1)

    def add_nodes(
        self, indices: ArrayLike, is_first_t: ArrayLike, is_last_t: ArrayLike
    ) -> None:
        """Add nodes to heuristic model.

        Parameters
        ----------
        indices : ArrayLike
            Nodes indices.
        is_first_t : ArrayLike
            Boolean array indicating if it belongs to first time point and it won't receive appearance penalization.
        is_last_t : ArrayLike
            Boolean array indicating if it belongs to last time point and it won't receive disappearance penalization.
        """
        if hasattr(self, "_forbidden"):
            raise ValueError("Nodes have already been added.")

        assert_same_length(indices=indices, is_first_t=is_first_t, is_last_t=is_last_t)

        indices = np.asarray(indices)
        size = len(indices)
        self._forward_map = ArrayMap(indices, np.arange(size))
        self._backward_map = indices.copy()

        self._appear_weight = np.asarray(
            np.logical_not(is_first_t) * self._config.appear_weight,
            dtype=np.float32,
        )
        self._disappear_weight = np.asarray(
            np.logical_not(is_last_t) * self._config.disappear_weight,
            dtype=np.float32,
        )

        self._forbidden = np.zeros(size, dtype=bool)
        self._in_count = np.zeros(size, dtype=np.uint8)
        self._out_count = np.zeros(size, dtype=np.uint8)
        self._selected_nodes = np.zeros(size, dtype=bool)

        self._predecessor_map = np.full(size, -1, dtype=np.int64)
        self._predecessor_weight = np.full(size, 1e6, dtype=np.float32)

    def add_edges(
        self, sources: ArrayLike, targets: ArrayLike, weights: ArrayLike
    ) -> None:
        """Add edges to model and applies weights link function from config.

        Parameters
        ----------
        source : ArrayLike
            Array of integers indicating source indices.
        targets : ArrayLike
            Array of integers indicating target indices.
        weights : ArrayLike
            Array of weights, input to the link function.
        """
        if hasattr(self, "_weights"):
            raise ValueError("Edges have already been added.")

        assert_same_length(weights=weights, sources=sources, targets=targets)

        self._weights = np.asarray(
            self._config.apply_link_function(weights), np.float32
        )
        self._out_edge = self._forward_map[np.asarray(sources)].astype(np.int64)
        self._in_edge = self._forward_map[np.asarray(targets)].astype(np.int64)

        LOG.info("transformed edge weights %s", self._weights)

        n_nodes = len(self._appear_weight)

        self._in_out_digraph = sparse.csr_matrix(
            (self._weights, (self._in_edge, self._out_edge)),
            shape=(n_nodes, n_nodes),
            dtype=np.float32,
        )

    def add_overlap_constraints(self, sources: ArrayLike, targets: ArrayLike) -> None:
        """Add constraints such that `source` and `target` can't be present in the same solution.

        Parameters
        ----------
        sources : ArrayLike
            Source nodes indices.
        targets : ArrayLike
            Target nodes indices.
        """
        sources = self._forward_map[np.asarray(sources)]
        targets = self._forward_map[np.asarray(targets)]
        mask = np.ones(len(sources), dtype=bool)
        size = len(self._appear_weight)
        self._overlap = sparse.csr_matrix(
            (mask, (sources, targets)), shape=(size, size), dtype=bool
        )

    def enforce_nodes_solution_value(self, indices: ArrayLike) -> None:
        """Constraints given nodes' variables to 1.

        Parameters
        ----------
        indices : ArrayLike
            Nodes indices.
        """
        if len(indices) == 0:
            return

        indices = self._forward_map[np.asarray(indices)]
        for i in indices:
            self._forbid_overlap(i)

    def _forbid_overlap(self, node_index: int) -> None:
        """Mark nodes overlapping with the given index as forbidden."""
        row = self._overlap.indices[
            self._overlap.indptr[node_index] : self._overlap.indptr[node_index + 1]
        ]
        self._forbidden[row] = True

    def optimize(self) -> float:
        """Optimizes objective function."""

        solver = NumbaHeuristicSolver(
            weights=self._weights,
            in_edge=self._in_edge,
            out_edge=self._out_edge,
            forbidden=self._forbidden,
            in_count=self._in_count,
            out_count=self._out_count,
            selected_nodes=self._selected_nodes,
            predecessor_map=self._predecessor_map,
            predecessor_weight=self._predecessor_weight,
            overlap_indices=self._overlap.indices,
            overlap_indptr=self._overlap.indptr,
            in_out_digraph_indices=self._in_out_digraph.indices,
            in_out_digraph_indptr=self._in_out_digraph.indptr,
            in_out_digraph_data=self._in_out_digraph.data,
            appear_weight=self._appear_weight,
            disappear_weight=self._disappear_weight,
            division_weight=self._config.division_weight,
            add_in_map=self._add_in_map,
            sub_in_map=self._sub_in_map,
            add_out_map=self._add_out_map,
            sub_out_map=self._sub_out_map,
        )
        solver.optimize()

        return solver._objective

    def solution(self) -> pd.DataFrame:
        """Returns the nodes present on the solution.

        Returns
        -------
        pd.DataFrame
            Dataframe indexed by nodes as indices and their parent (NA if orphan).
        """
        nodes = self._backward_map[np.nonzero(self._selected_nodes)[0]]

        nodes = pd.DataFrame(
            data=NO_PARENT,
            index=nodes,
            columns=["parent_id"],
        )

        (node_id,) = np.nonzero(self._predecessor_map != -1)  # 0 index space
        parent_id = self._backward_map[self._predecessor_map[node_id]]
        node_id = self._backward_map[node_id]  # original space

        inv_edges = pd.DataFrame(
            data=parent_id,
            index=node_id,
            columns=["parent_id"],
        )

        nodes.update(inv_edges)
        return nodes
