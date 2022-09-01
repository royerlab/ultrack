import logging

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from scipy import sparse
from skimage.util._map_array import ArrayMap

from ultrack.config.config import TrackingConfig
from ultrack.core.database import NO_PARENT
from ultrack.core.tracking.solver.base_solver import BaseSolver
from ultrack.core.tracking.solver.heuristic._heap import Heap

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

        self._assert_same_length(
            indices=indices, is_first_t=is_first_t, is_last_t=is_last_t
        )

        size = len(indices)
        self._forward_map = ArrayMap(indices, np.arange(size))
        self._backward_map = indices.copy()

        self._appear_weight = np.logical_not(is_first_t) * self._config.appear_weight
        self._disappear_weight = (
            np.logical_not(is_last_t) * self._config.disappear_weight
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

        self._assert_same_length(weights=weights, sources=sources, targets=targets)

        self._weights = self._config.apply_link_function(weights).astype(np.float32)
        self._out_edge = self._forward_map[np.asarray(sources)]
        self._in_edge = self._forward_map[np.asarray(targets)]

        LOG.info(f"transformed edge weights {self._weights}")

        n_nodes = len(self._appear_weight)

        self._in_out_digraph = sparse.csr_matrix(
            (self._weights, (self._in_edge, self._out_edge)),
            shape=(n_nodes, n_nodes),
            dtype=np.float32,
        )

    def add_overlap_constraints(self, source: ArrayLike, target: ArrayLike) -> None:
        """Add constraints such that `source` and `target` can't be present in the same solution.

        Parameters
        ----------
        source : ArrayLike
            Source nodes indices.
        target : ArrayLike
            Target nodes indices.
        """
        source = self._forward_map[np.asarray(source)]
        target = self._forward_map[np.asarray(target)]
        mask = np.ones(len(source), dtype=bool)
        size = len(self._appear_weight)
        self._overlap = sparse.csr_matrix(
            (mask, (source, target)), shape=(size, size), dtype=bool
        )

    def enforce_node_to_solution(self, indices: ArrayLike) -> None:
        """Constraints given nodes' variables to 1.

        Parameters
        ----------
        indices : ArrayLike
            Nodes indices.
        """
        indices = self._forward_map[np.asarray(indices)]
        for i in indices:
            self._forbid_overlap(i)

    def optimize(self) -> float:
        """Optimizes objective function."""

        ascent_iters = 5
        self.mst_pass()

        for _ in range(ascent_iters):
            self.random_ascent()

        return self._objective

    def _transition_weight(self, transition: np.ndarray, node_id: int) -> float:
        """Computes the cost of a given transition for the given node.

        Parameters
        ----------
        transition : np.ndarray
            (2, 3, 3) transition matrix.
        node_id : int
            Node index.

        Returns
        -------
        float
            The cost (weight) of the transition.
        """
        weights = (
            self._appear_weight[node_id],
            self._disappear_weight[node_id],
            self._config.division_weight,
        )
        return np.sum(
            transition[self._in_count[node_id], self._out_count[node_id]] * weights
        )

    def mst_pass(self) -> float:
        """
        Executes Kruskal MST algorithm while preserving the biological constraints.
        """
        heap = Heap(self._weights)
        heap.insert_array(np.arange(len(self._weights)))

        while not heap.is_empty():
            edge_index = heap.pop()
            in_node = self._in_edge[edge_index]
            out_node = self._out_edge[edge_index]

            # blocked by overlap
            if self._forbidden[in_node] or self._forbidden[out_node]:
                continue

            # check for flow constrains
            if self._in_count[in_node] > 0 or self._out_count[out_node] > 1:
                continue

            weight = self._weights[edge_index]
            obj_delta = weight
            obj_delta += self._transition_weight(self._add_in_map, in_node)
            obj_delta += self._transition_weight(self._add_out_map, out_node)

            # any other check?
            self._objective += obj_delta

            self._selected_nodes[out_node] = True
            self._selected_nodes[in_node] = True
            self._predecessor_map[in_node] = out_node
            self._predecessor_weight[in_node] = weight

            self._out_count[out_node] += 1
            self._in_count[in_node] += 1
            self._forbid_overlap(in_node)
            self._forbid_overlap(out_node)

    def _local_search(self, node_index: int) -> float:
        """Performs a local search over the predecessors (out_node) of the given node.

        Parameters
        ----------
        node_index : int
            Node index.

        Returns
        -------
        float
            Change in objective function.
        """
        objective = 0.0
        current_out_node = self._predecessor_map[node_index]
        if current_out_node != -1:
            # partial removal of current node
            objective += (
                self._transition_weight(self._sub_out_map, current_out_node)
                - self._predecessor_weight[node_index]
            )
            self._selected_nodes[current_out_node] = False
            self._out_count[current_out_node] -= 1
            self._release_overlap(current_out_node)
        else:
            # add in node if it isn't in solution
            objective + self._transition_weight(self._add_in_map, node_index)

        # search local maximum
        max_obj_delta = 0.0
        argmax_obj_delta = 0.0
        argmax_weight = 0.0
        for i in range(
            self._in_out_digraph.indptr[node_index],
            self._in_out_digraph.indptr[node_index + 1],
        ):
            out_node = self._in_out_digraph.indices[i]
            if self._out_count[out_node] > 1 or self._forbidden[out_node]:
                continue

            weight = self._in_out_digraph.data[i]
            delta = self._transition_weight(self._add_out_map, out_node) + weight
            if delta > max_obj_delta:
                max_obj_delta = delta
                argmax_obj_delta = out_node
                argmax_weight = weight

        if objective + max_obj_delta <= 0.0:  # can't improve solution
            # return previous out node to solution
            if current_out_node != -1:
                self._forbid_overlap(current_out_node)
                self._selected_nodes[current_out_node] = True
                self._out_count[current_out_node] += 1

            return 0.0

        if current_out_node == -1:
            # add in node to solution
            self._in_count[node_index] += 1
            self._forbid_overlap(node_index)
            self._selected_nodes[node_index] = True

        # adding to solution
        self._forbid_overlap(argmax_obj_delta)
        self._selected_nodes[argmax_obj_delta] = True
        self._out_count[argmax_obj_delta] += 1
        self._predecessor_map[node_index] = argmax_obj_delta
        self._predecessor_weight[node_index] = argmax_weight

        return objective + max_obj_delta

    def random_ascent(self) -> None:
        """Executes random ascent on the current solution."""
        n_nodes = len(self._appear_weight)
        node_ids = self._rng.choice(n_nodes, size=n_nodes, replace=False)

        for node_id in node_ids:
            self._objective += self._local_search(node_id)

    def _forbid_overlap(self, node_index: int) -> None:
        """Mark nodes overlapping with the given index as forbidden."""
        row = self._overlap.indices[
            self._overlap.indptr[node_index] : self._overlap.indptr[node_index + 1]
        ]
        self._forbidden[row] = True

    def _release_overlap(self, node_index: int) -> None:
        """Releases forbidden mark of nodes overlappping with the given index."""
        row = self._overlap.indices[
            self._overlap.indptr[node_index] : self._overlap.indptr[node_index + 1]
        ]
        for i in row:
            indices = self._overlap.indices[
                self._overlap.indptr[i] : self._overlap.indptr[i + 1]
            ]
            self._forbidden[i] = np.any(self._selected_nodes[indices])

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
