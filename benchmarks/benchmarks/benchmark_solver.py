from typing import Dict, Tuple

import numpy as np

from ultrack.config import TrackingConfig
from ultrack.core.solve.solver import HeuristicSolver, MIPSolver


class SolverSuite:
    params = [
        ["mip", "heuristic"],
        [
            (10, 100, 5, 5),
            (10, 1_000, 5, 5),
            (20, 1_000, 5, 5),
            (20, 1_000, 10, 10),
            (50, 10_000, 10, 10),
        ],
    ]

    timeout = 3600  # 1 hour timeout

    # params = length, n_nodes_per_time, n_neighbors, n_overlaps
    param_names = ["solver_class", "params"]

    _SOLVER_CLASS = {"mip": MIPSolver, "heuristic": HeuristicSolver}

    def setup(self, solver: str, params: Tuple[int]) -> None:
        length, n_nodes_per_time, n_neighbors, n_overlaps = params
        self._config = TrackingConfig()
        self._rng = np.random.default_rng(42)
        self._add_nodes_data = self._sequential_nodes(length, n_nodes_per_time)
        self._add_edges_data = self._random_pairs(
            length, n_nodes_per_time, n_neighbors, shift_over_time=True
        )
        self._add_edges_data["weights"] = self._rng.uniform(
            low=-0.25, high=1, size=len(self._add_edges_data["sources"])
        )
        self._add_overlaps_data = self._random_pairs(
            length, n_nodes_per_time, n_overlaps, shift_over_time=False
        )
        self._solver = self.time_initialization(solver)

    @staticmethod
    def _sequential_nodes(length: int, n_nodes_per_time: int) -> Dict[str, np.ndarray]:
        node_indices = np.arange(length * n_nodes_per_time)
        time = np.repeat(np.arange(length), n_nodes_per_time)
        return {
            "indices": node_indices,
            "is_first_t": time == 0,
            "is_last_t": time == length - 1,
        }

    def _random_pairs(
        self,
        length: int,
        n_nodes_per_time: int,
        n_neighbors: int,
        shift_over_time: bool,
    ) -> Dict[str, np.ndarray]:
        sources = []
        targets = []
        for i in range(length - shift_over_time):
            start_id = i * n_nodes_per_time
            source = np.repeat(
                np.arange(start_id, start_id + n_nodes_per_time), n_neighbors
            )

            start_id = i * (n_nodes_per_time + shift_over_time)
            target = self._rng.integers(
                low=start_id,
                high=start_id + n_nodes_per_time,
                size=(n_nodes_per_time * n_neighbors),
            )

            sources.append(source)
            targets.append(target)

        sources = np.concatenate(sources)
        targets = np.concatenate(targets)
        if not shift_over_time:
            # avoiding self linkaged
            targets = np.where(
                sources == targets,
                np.mod(targets + 1, n_nodes_per_time * length),
                targets,
            )
        return {"sources": sources, "targets": targets}

    def time_initialization(self, solver: str, *params) -> HeuristicSolver:
        solver = self._SOLVER_CLASS[solver](self._config)
        solver.add_nodes(**self._add_nodes_data)
        solver.add_edges(**self._add_edges_data)
        solver.add_overlap_constraints(**self._add_overlaps_data)
        return solver

    def time_optimize(self, *params) -> None:
        self._solver.optimize()

    def track_objective(self, *params) -> float:
        return self._solver.optimize()
