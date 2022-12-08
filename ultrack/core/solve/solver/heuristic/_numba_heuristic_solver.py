import numba
import numpy as np
from numba.experimental import jitclass

from ultrack.core.solve.solver.heuristic._heap import Heap, Policy

_SPEC = {
    "_objective": numba.float64,
    "_weights": numba.float32[:],
    "_in_edge": numba.int64[:],
    "_out_edge": numba.int64[:],
    "_forbidden": numba.boolean[:],
    "_in_count": numba.uint8[:],
    "_out_count": numba.uint8[:],
    "_selected_nodes": numba.boolean[:],
    "_predecessor_map": numba.int64[:],
    "_predecessor_weight": numba.float32[:],
    "_overlap_indices": numba.int32[:],
    "_overlap_indptr": numba.int32[:],
    "_in_out_digraph_indices": numba.int32[:],
    "_in_out_digraph_indptr": numba.int32[:],
    "_in_out_digraph_data": numba.float32[:],
    "_appear_weight": numba.float32[:],
    "_disappear_weight": numba.float32[:],
    "_division_weight": numba.float64,
    "_add_in_map": numba.float32[:, :, :],
    "_sub_in_map": numba.float32[:, :, :],
    "_add_out_map": numba.float32[:, :, :],
    "_sub_out_map": numba.float32[:, :, :],
}


@jitclass(_SPEC)
class NumbaHeuristicSolver:
    def __init__(
        self,
        weights: np.ndarray,
        in_edge: np.ndarray,
        out_edge: np.ndarray,
        forbidden: np.ndarray,
        in_count: np.ndarray,
        out_count: np.ndarray,
        selected_nodes: np.ndarray,
        predecessor_map: np.ndarray,
        predecessor_weight: np.ndarray,
        overlap_indices: np.ndarray,
        overlap_indptr: np.ndarray,
        in_out_digraph_indices: np.ndarray,
        in_out_digraph_indptr: np.ndarray,
        in_out_digraph_data: np.ndarray,
        appear_weight: np.ndarray,
        disappear_weight: np.ndarray,
        division_weight: float,
        add_in_map: np.ndarray,
        sub_in_map: np.ndarray,
        add_out_map: np.ndarray,
        sub_out_map: np.ndarray,
    ) -> None:

        self._objective = 0.0
        self._weights = weights
        self._in_edge = in_edge
        self._out_edge = out_edge
        self._forbidden = forbidden
        self._in_count = in_count
        self._out_count = out_count
        self._selected_nodes = selected_nodes
        self._predecessor_map = predecessor_map
        self._predecessor_weight = predecessor_weight
        self._overlap_indices = overlap_indices
        self._overlap_indptr = overlap_indptr
        self._in_out_digraph_indices = in_out_digraph_indices
        self._in_out_digraph_indptr = in_out_digraph_indptr
        self._in_out_digraph_data = in_out_digraph_data
        self._appear_weight = appear_weight
        self._disappear_weight = disappear_weight
        self._division_weight = division_weight
        self._add_in_map = add_in_map
        self._sub_in_map = sub_in_map
        self._add_out_map = add_out_map
        self._sub_out_map = sub_out_map

    def optimize(self) -> float:
        """Optimizes objective function."""
        np.random.seed(42)

        ascent_iters = 5
        obj_tol = 1
        self.constrained_mst()

        prev_objective = 0.0
        for _ in range(ascent_iters):
            self.random_ascent()
            if prev_objective - self._objective < obj_tol:
                break
            prev_objective = self._objective

        return self._objective

    def constrained_mst(self) -> None:
        """
        Executes Kruskal MST algorithm while preserving the biological constraints.
        """
        heap = Heap(self._weights, Policy.maximum)
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

    def random_ascent(self) -> None:
        """Executes random ascent on the current solution."""
        n_nodes = len(self._appear_weight)
        node_ids = np.random.choice(n_nodes, size=n_nodes, replace=False)

        for node_id in node_ids:
            self._objective += self._local_search(node_id)

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
        max_obj_delta = np.finfo(np.float32).min
        argmax_obj_delta = -1
        argmax_weight = 0.0
        for i in range(
            self._in_out_digraph_indptr[node_index],
            self._in_out_digraph_indptr[node_index + 1],
        ):
            out_node = self._in_out_digraph_indices[i]
            if self._out_count[out_node] > 1 or self._forbidden[out_node]:
                continue

            weight = self._in_out_digraph_data[i]
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

        # sanity check
        assert argmax_obj_delta != -1

        # adding to solution
        self._forbid_overlap(argmax_obj_delta)
        self._selected_nodes[argmax_obj_delta] = True
        self._out_count[argmax_obj_delta] += 1
        self._predecessor_map[node_index] = argmax_obj_delta
        self._predecessor_weight[node_index] = argmax_weight

        return objective + max_obj_delta

    def _forbid_overlap(self, node_index: int) -> None:
        """Mark nodes overlapping with the given index as forbidden."""
        row = self._overlap_indices[
            self._overlap_indptr[node_index] : self._overlap_indptr[node_index + 1]
        ]
        self._forbidden[row] = True

    def _release_overlap(self, node_index: int) -> None:
        """Releases forbidden mark of nodes overlappping with the given index."""
        row = self._overlap_indices[
            self._overlap_indptr[node_index] : self._overlap_indptr[node_index + 1]
        ]
        for i in row:
            indices = self._overlap_indices[
                self._overlap_indptr[i] : self._overlap_indptr[i + 1]
            ]
            self._forbidden[i] = np.any(self._selected_nodes[indices])

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
        weights = np.asarray(
            [
                self._appear_weight[node_id],
                self._disappear_weight[node_id],
                self._division_weight,
            ]
        )
        return np.sum(
            transition[self._in_count[node_id], self._out_count[node_id]] * weights
        )
