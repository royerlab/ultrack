import numpy as np
import pandas as pd
import pytest

from ultrack.config.config import MainConfig
from ultrack.core.solve.solver.mip_solver import MIPSolver
from ultrack.core.solve.sqltracking import _check_inside_border


@pytest.mark.parametrize(
    "config_content",
    [
        {
            "tracking.appear_weight": -0.25,
            "tracking.disappear_weight": -1.0,
            "tracking.division_weight": -0.5,
            "tracking.link_function": "identity",
            "tracking.bias": 0,
        }
    ],
    indirect=True,
)
def test_solvers_optimize(config_instance: MainConfig) -> None:
    """
    This demo builds a very simple graph with 7 nodes and a single overlap constraint (2,5)
    and a two possible divisions on 2 and 6.

    Edge -C- denotes contraint.

    Graph:

    1 - 0.5 - 2 - 0.5 - 3 - 0.5 - 4
              |  \\             /      \\ due linting software
              C   1.0      0.9
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
              C   1.0      0.9
              |       \\  /
              5 - 0.5 - 6 - 0.7 - 7
                        ^
                        |
            THIS WILL IS FIXED AS DIVISION

    Solution:

    1 - 0.5 - 2                   4
                \\              /
                  1.0      0.9
                     \\  /
                       6 - 0.7 - 7

    Result: 0.5 + 1.0 + 0.7 + 0.9 - division_weight
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


def test_solver_with_node_probabilities(config_instance: MainConfig) -> None:
    """
    Edge -C- denotes contraint.

    Graph:

   0.3       0.7       1.0       1.0
    1 - 0.5 - 2 - 0.5 - 3 - 0.5 - 4
              |  \\             /      \\ due linting software
              C   1.0      0.9
              |       \\  /
              5 - 0.5 - 6 - 0.7 - 7
    node w.  0.5       1.0       1.0

    Solution:

   0.3       0.7       1.0       1.0
    1 - 0.5 - 2 - 0.5 - 3 - 0.5 - 4
                 \\
                  1.0
                      \\
                        6 - 0.7 - 7
    node w.  0.5       1.0       1.0

    Result: 0.3 + 0.7 + 1.0 + 1.0 + 1.0 + 1.0 +
            0.5 + 0.5 + 0.5 + 1.0 + 0.7 - division_weight
    """
    solver = MIPSolver(config_instance.tracking_config)

    nodes = np.array([1, 2, 3, 4, 5, 6, 7])
    nodes_probs = np.array([0.3, 0.7, 1.0, 1.0, 0.5, 1.0, 1.0])
    is_first = np.array([1, 0, 0, 0, 0, 0, 0], dtype=bool)
    is_last = np.array([0, 0, 0, 1, 0, 0, 1], dtype=bool)

    solver.add_nodes(nodes, is_first, is_last, nodes_prob=nodes_probs)

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
        nodes_probs[expected_solution.index.to_numpy() - 1].sum()
        + weights[expected_edges].sum()
        + config_instance.tracking_config.division_weight,
    )


def test_fixed_edges_constraint_solver(config_instance: MainConfig) -> None:
    """
    Same graph as before but with a fixed division on node 6.

    Edge -C- denotes contraint.

    Graph:

    1 - 0.5 - 2 - 0.5 - 3 - 0.5 - 4
              |  \\             /      \\ due linting software
              C   1.0      0.9
              |       \\  /
              5 - 0.5 - 6 - 0.7 - 7
                             ^
                             |
                        FORBIDDEN EDGE

    Solution:

    1 - 0.5 - 2                   4
                \\              /
                  1.0      0.9
                     \\  /
                       6

    Result: 0.5 + 1.0 + 0.9
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


@pytest.mark.parametrize(
    "config_content",
    [
        {
            "tracking.image_border_size": (1, 3, 2),
        },
    ],
    indirect=True,
)
def test_add_nodes_with_is_border(config_instance: MainConfig):
    """
    Test the add_nodes method with the is_border parameter.

    Assuming TZYX shape = (10,10,10,10) and varying image_border_size
    Testing if node is inside the border in all combinations of z, y, x
    """
    solver = MIPSolver(config_instance.tracking_config)

    shape = (10, 10, 10, 10)  # TZYX

    data = {
        "id": np.arange(10),
        "t": np.concatenate([np.zeros(5), np.ones(5)]),
        "z": np.array([0, 0, 0, 0, 5, 0, 5, 9, 5, 9]),
        "y": np.array([5, 2, 2, 0, 5, 5, 2, 9, 5, 5]),
        "x": np.array([5, 5, 5, 0, 1, 1, 1, 9, 5, 9]),
    }
    df = pd.DataFrame(data)
    is_border = _check_inside_border(
        df, config_instance.tracking_config.image_border_size, shape=shape
    )

    # Assert that the is_border Series is as expected
    expected_is_border = np.asarray(
        [True, True, True, True, True, True, True, True, False, True]
    )
    np.testing.assert_array_equal(is_border, expected_is_border)

    # Add nodes to the solver
    solver.add_nodes(
        df["id"],
        # avoiding first and last nodes zero (dist)appearance cost
        np.zeros(10, dtype=bool),
        np.zeros(10, dtype=bool),
        is_border=is_border,
    )

    # Assertions to verify the nodes were added correctly
    assert solver._nodes is not None
    assert solver._appearances is not None
    assert solver._disappearances is not None
    assert solver._divisions is not None

    # Verify that the is_border values are correctly processed
    for idx, border in enumerate(is_border):
        if border:
            # Verify that appearance and disappearance weights are set to 0 for border nodes
            assert solver._appearances[idx].obj == 0
            assert solver._disappearances[idx].obj == 0
        else:
            assert (
                solver._appearances[idx].obj
                == config_instance.tracking_config.appear_weight
            )
            assert (
                solver._disappearances[idx].obj
                == config_instance.tracking_config.disappear_weight
            )
