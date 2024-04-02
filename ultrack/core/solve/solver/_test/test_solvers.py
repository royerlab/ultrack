from itertools import product

import numpy as np
import pandas as pd
import pytest

from ultrack.config.config import MainConfig
from ultrack.core.solve.solver.base_solver import BaseSolver
from ultrack.core.solve.solver.heuristic.heuristic_solver import HeuristicSolver
from ultrack.core.solve.solver.mip_solver import MIPSolver


@pytest.mark.parametrize(
    "solver,config_content",
    list(
        product(
            [MIPSolver, HeuristicSolver],
            [
                {
                    "tracking.appear_weight": -0.25,
                    "tracking.disappear_weight": -1.0,
                    "tracking.division_weight": -0.5,
                    "tracking.link_function": "identity",
                    "tracking.bias": 0,
                }
            ],
        )
    ),
    indirect=["config_content"],
)
def test_solvers_optimize(solver: BaseSolver, config_instance: MainConfig) -> None:
    """
    This demo builds a very simple graph with 7 nodes and a single overlap constraint (2,5)
    and a two possible divisions on 2 and 6.

    Edge -C- denotes contraint.

    Graph:

    1 - 0.5 - 2 - 0.5 - 3 - 0.5 - 4
              |  \\             /      \\ due linting software
              C   1.0      0.95
              |       \\  /
              5 - 0.5 - 6 - 0.7 - 7

    Solution:

    1 - 0.5 - 2 - 0.5 - 3 - 0.5 - 4
                \
                  1.0
                     \
                       6 - 0.7 - 7

    Result: 0.5 + 0.5 + 1.0 + 0.7 - division_weight
    """
    solver = solver(config_instance.tracking_config)

    nodes = np.array([1, 2, 3, 4, 5, 6, 7])
    is_first = np.array([1, 0, 0, 0, 0, 0, 0], dtype=bool)
    is_last = np.array([0, 0, 0, 1, 0, 0, 1], dtype=bool)

    solver.add_nodes(nodes, is_first, is_last)

    edges = np.array([[1, 2], [2, 3], [2, 6], [3, 4], [5, 6], [6, 4], [6, 7]])

    weights = np.array([0.5, 0.5, 1.0, 0.5, 0.5, 0.9, 0.7])
    solver.add_edges(edges[:, 0], edges[:, 1], weights)

    solver.set_standard_constraints()

    solver.add_overlap_constraints([2], [5])

    objective = solver.optimize()
    solution = solver.solution()

    expected_solution = pd.DataFrame(
        data=[pd.NA, 1, 2, 3, 2, 6],
        index=[1, 2, 3, 4, 6, 7],
        columns=["parent_id"],
        dtype=pd.Int64Dtype(),
    )
    expected_edges = np.array([1, 1, 1, 1, 0, 0, 1], dtype=bool)

    assert solution.shape == expected_solution.shape
    assert np.all(solution.index.isin(expected_solution.index))
    assert np.all(
        expected_solution.loc[solution.index, "parent_id"] == solution["parent_id"]
    )
    assert np.allclose(
        objective,
        weights[expected_edges].sum() + config_instance.tracking_config.division_weight,
    )


def test_fixed_nodes_constraint_solver(config_instance: MainConfig) -> None:
    """
    Same graph as before but with a fixed division on node 6.

    Edge -C- denotes contraint.

    Graph:

    1 - 0.5 - 2 - 0.5 - 3 - 0.5 - 4
              |  \\             /      \\ due linting software
              C   1.0      0.95
              |       \\  /
              5 - 0.5 - 6 - 0.7 - 7
                        ^
                        |
            THIS WILL IS FIXED AS DIVISION

    Solution:

    1 - 0.5 - 2                   4
                \\              /
                  1.0      0.95
                     \\  /
                       6 - 0.7 - 7

    Result: 0.5 + 1.0 + 0.7 + 0.95 - division_weight
    """
    solver = MIPSolver(config_instance.tracking_config)

    nodes = np.array([1, 2, 3, 4, 5, 6, 7])
    is_first = np.array([1, 0, 0, 0, 0, 0, 0], dtype=bool)
    is_last = np.array([0, 0, 0, 1, 0, 0, 1], dtype=bool)

    solver.add_nodes(nodes, is_first, is_last)

    edges = np.array([[1, 2], [2, 3], [2, 6], [3, 4], [5, 6], [6, 4], [6, 7]])

    weights = np.array([0.5, 0.5, 1.0, 0.5, 0.5, 0.9, 0.7])
    solver.add_edges(edges[:, 0], edges[:, 1], weights)

    solver.set_standard_constraints()

    solver.add_overlap_constraints([2], [5])

    solver.enforce_nodes_solution_value([6], "division", True)

    objective = solver.optimize()
    solution = solver.solution()

    expected_solution = pd.DataFrame(
        data=[pd.NA, 1, 6, 2, 6],
        index=[1, 2, 4, 6, 7],
        columns=["parent_id"],
        dtype=pd.Int64Dtype(),
    )
    expected_edges = np.array([1, 0, 1, 0, 0, 1, 1], dtype=bool)

    assert solution.shape == expected_solution.shape
    assert np.all(solution.index.isin(expected_solution.index))
    assert np.all(
        expected_solution.loc[solution.index, "parent_id"] == solution["parent_id"]
    )
    assert np.allclose(
        objective,
        weights[expected_edges].sum() + config_instance.tracking_config.division_weight,
    )


def test_fixed_edges_constraint_solver(config_instance: MainConfig) -> None:
    """
    Same graph as before but with a fixed division on node 6.

    Edge -C- denotes contraint.

    Graph:

    1 - 0.5 - 2 - 0.5 - 3 - 0.5 - 4
              |  \\             /      \\ due linting software
              C   1.0      0.95
              |       \\  /
              5 - 0.5 - 6 - 0.7 - 7
                             ^
                             |
                        FORBIDDEN EDGE

    Solution:

    1 - 0.5 - 2                   4
                \\              /
                  1.0      0.95
                     \\  /
                       6

    Result: 0.5 + 1.0 + 0.95
    """
    solver = MIPSolver(config_instance.tracking_config)

    nodes = np.array([1, 2, 3, 4, 5, 6, 7])
    is_first = np.array([1, 0, 0, 0, 0, 0, 0], dtype=bool)
    is_last = np.array([0, 0, 0, 1, 0, 0, 1], dtype=bool)

    solver.add_nodes(nodes, is_first, is_last)

    edges = np.array([[1, 2], [2, 3], [2, 6], [3, 4], [5, 6], [6, 4], [6, 7]])

    weights = np.array([0.5, 0.5, 1.0, 0.5, 0.5, 0.9, 0.7])
    solver.add_edges(edges[:, 0], edges[:, 1], weights)

    solver.set_standard_constraints()

    solver.add_overlap_constraints([2], [5])

    solver.enforce_edges_solution_value([6], [7], False)

    objective = solver.optimize()
    solution = solver.solution()

    expected_solution = pd.DataFrame(
        data=[pd.NA, 1, 6, 2],
        index=[1, 2, 4, 6],
        columns=["parent_id"],
        dtype=pd.Int64Dtype(),
    )
    expected_edges = np.array([1, 0, 1, 0, 0, 1, 0], dtype=bool)

    assert solution.shape == expected_solution.shape
    assert np.all(solution.index.isin(expected_solution.index))
    assert np.all(
        expected_solution.loc[solution.index, "parent_id"] == solution["parent_id"]
    )
    assert np.allclose(objective, weights[expected_edges].sum())
