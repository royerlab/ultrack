import logging

import gurobipy as gp
import numpy as np
import pandas as pd
from gurobipy import GRB
from numpy.typing import ArrayLike

from ultrack.config.config import TrackingConfig
from ultrack.core.database import NO_PARENT
from ultrack.core.solve.solver.base_solver import BaseSolver

LOG = logging.getLogger(__name__)


class GurobiSolver(BaseSolver):
    def __init__(
        self,
        config: TrackingConfig,
    ) -> None:
        """Generic maxflow gurobi solver for cell-tracking ILP.

        Parameters
        ----------
        config : TrackingConfig
            Tracking configuration parameters.
        """

        self._config = config
        self.reset()

    def reset(self) -> None:
        """Sets model to an empty state."""
        self._model = gp.Model()
        self._nodes = []
        self._edges = {}
        self._appearances = {}
        self._disappearances = {}
        self._divisions = {}

        self._setup_model_parameters()

    def _setup_model_parameters(self) -> None:
        """Sets model parameters from configuration file."""
        self._model.ModelSense = GRB.MAXIMIZE
        self._model.Params.TimeLimit = self._config.time_limit
        self._model.Params.Threads = self._config.n_threads
        self._model.Params.Method = self._config.method
        self._model.Params.MIPGap = self._config.solution_gap

    def add_nodes(
        self, indices: ArrayLike, is_first_t: ArrayLike, is_last_t: ArrayLike
    ) -> None:
        """Add nodes slack variables to gurobi model.

        Parameters
        ----------
        indices : ArrayLike
            Nodes indices.
        is_first_t : ArrayLike
            Boolean array indicating if it belongs to first time point and it won't receive appearance penalization.
        is_last_t : ArrayLike
            Boolean array indicating if it belongs to last time point and it won't receive disappearance penalization.
        """
        if len(self._nodes) > 0:
            raise ValueError("Nodes have already been added.")

        self._assert_same_length(
            indices=indices, is_first_t=is_first_t, is_last_t=is_last_t
        )

        LOG.info(f"# {np.sum(is_first_t)} nodes at starting `t`.")
        LOG.info(f"# {np.sum(is_last_t)} nodes at last `t`.")

        appear_weight = np.logical_not(is_first_t) * self._config.appear_weight
        disappear_weight = np.logical_not(is_last_t) * self._config.disappear_weight

        indices = indices.tolist()
        self._nodes = indices  # self._model.addVars(indices, vtype=GRB.BINARY)
        self._appearances = self._model.addVars(
            indices, vtype=GRB.BINARY, obj=appear_weight.tolist()
        )
        self._disappearances = self._model.addVars(
            indices, vtype=GRB.BINARY, obj=disappear_weight.tolist()
        )
        self._divisions = self._model.addVars(
            indices, vtype=GRB.BINARY, obj=self._config.division_weight
        )

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
        if len(self._edges) > 0:
            raise ValueError("Edges have already been added.")

        self._assert_same_length(sources=sources, targets=targets, weights=weights)

        weights = self._config.apply_link_function(weights)

        LOG.info(f"transformed edge weights {weights}")

        variables = {(s, t): w for s, t, w in zip(sources, targets, weights)}
        self._edges = self._model.addVars(
            variables.keys(), obj=variables, vtype=GRB.BINARY
        )

    def set_standard_constraints(self) -> None:
        """Sets standard biological/flow constraints:
        - single incoming node (no fusion);
        - flow conservation (begin and end requires slack variables);
        - divisions only from existing nodes;
        """

        # flow conservation
        self._model.addConstrs(
            self._edges.sum("*", i) + self._appearances[i] + self._divisions[i]
            == self._edges.sum(i, "*") + self._disappearances[i]
            for i in self._nodes
        )

        # divisions
        self._model.addConstrs(
            self._edges.sum("*", i) + self._appearances[i] >= self._divisions[i]
            for i in self._nodes
        )

    def add_overlap_constraints(self, sources: ArrayLike, targets: ArrayLike) -> None:
        """Add constraints such that `source` and `target` can't be present in the same solution.

        Parameters
        ----------
        source : ArrayLike
            Source nodes indices.
        target : ArrayLike
            Target nodes indices.
        """
        self._model.addConstrs(
            self._edges.sum(sources[i], "*") + self._edges.sum(targets[i], "*") <= 1
            for i in range(len(sources))
        )

    def enforce_node_to_solution(self, indices: ArrayLike) -> None:
        """Constraints given nodes' variables to 1.

        Parameters
        ----------
        indices : ArrayLike
            Nodes indices.
        """
        self._model.addConstrs(
            self._edges.sum(i, "*") + self._appearances[i] >= 1 for i in indices
        )

    def _set_solution_guess(self) -> None:
        # TODO
        pass

    def optimize(self) -> float:
        """Optimizes gurobi model."""
        self._model.optimize()
        return self._model.getObjective().getValue()

    def solution(self) -> pd.DataFrame:
        """Returns the nodes present on the solution.

        Returns
        -------
        pd.DataFrame
            Dataframe indexed by nodes as indices and their parent (NA if orphan).
        """
        if not (
            self._model.status == GRB.OPTIMAL or self._model.status == GRB.TIME_LIMIT
        ):
            raise ValueError(
                "Gurobi solver must be optimized before returning solution."
            )

        nodes = set()
        for k, var in self._edges.items():
            if var.X > 0.5:
                nodes.add(k[0])
                nodes.add(k[1])
        nodes = list(nodes)
        LOG.info(f"Solution nodes\n{nodes}")

        if len(nodes) == 0:
            raise ValueError("Something went wrong, nodes solution is empty.")

        nodes = pd.DataFrame(
            data=NO_PARENT,
            index=nodes,
            columns=["parent_id"],
        )

        inv_edges = np.asarray([k for k, var in self._edges.items() if var.X > 0.5])
        LOG.info(f"Solution edges\n{inv_edges}")

        if len(inv_edges) == 0:
            raise ValueError("Something went wrong, edges solution is empty")

        inv_edges = pd.DataFrame(
            data=inv_edges[:, 0],
            index=inv_edges[:, 1],
            columns=["parent_id"],
        )

        nodes.update(inv_edges)

        return nodes
