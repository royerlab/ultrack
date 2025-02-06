import logging
import uuid
from pathlib import Path
from typing import Literal, Optional

import mip
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from skimage.util._map_array import ArrayMap

from ultrack.config.config import TrackingConfig
from ultrack.core.solve.solver.base_solver import BaseSolver
from ultrack.utils.array import assert_same_length
from ultrack.utils.constants import NO_PARENT

LOG = logging.getLogger(__name__)

_KEY_TO_SOLVER_NAME = {
    "CBC": "Coin-OR Branch and Cut",
    "GRB": "Gurobi",
    "GUROBI": "Gurobi",
}


class MIPSolver(BaseSolver):
    def __init__(
        self,
        config: TrackingConfig,
    ) -> None:
        """Generic mixed-integer programming (MIP) solver for cell-tracking ILP.

        Parameters
        ----------
        config : TrackingConfig
            Tracking configuration parameters.
        """
        self._config = config
        self.reset()

    def reset(self) -> None:
        """Sets model to an empty state."""
        try:
            self._model = mip.Model(
                sense=mip.MAXIMIZE, solver_name=self._config.solver_name
            )
        except mip.exceptions.InterfacingError as e:
            LOG.warning(e)
            self._model = mip.Model(sense=mip.MAXIMIZE, solver_name=mip.CBC)

        print(f"Using {_KEY_TO_SOLVER_NAME[self._model.solver_name]} solver")

        if self._model.solver_name == mip.CBC:
            LOG.warning(
                "Using CBC solver. Consider installing Gurobi for significantly better performance."
            )
            LOG.warning(
                "To install Gurobi, follow the instructions at "
                "https://support.gurobi.com/hc/en-us/articles/360044290292-How-do-I-install-Gurobi-for-Python-"
            )
            LOG.warning(
                "It is free for academic use. "
                "See https://www.gurobi.com/academia/academic-program-and-licenses/"
            )

        self._forward_map = None
        self._backward_map = None
        self._nodes = None
        self._appearances = None
        self._disappearances = None
        self._divisions = None
        self._edges = None
        self._weights = None
        self._setup_model_parameters()

    def _setup_model_parameters(self) -> None:
        """Sets model parameters from configuration file."""
        self._model.max_seconds = self._config.time_limit
        self._model.threads = self._config.n_threads
        self._model.lp_method = self._config.method
        self._model.max_mip_gap = self._config.solution_gap

    def add_nodes(
        self,
        indices: ArrayLike,
        is_first_t: ArrayLike,
        is_last_t: ArrayLike,
        is_border: ArrayLike = False,
        nodes_prob: Optional[ArrayLike] = None,
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
        is_border : ArrayLike
            Boolean array indicating if it belongs to the border and it won't receive (dis)apperance penalization.
            Default: False
        nodes_prob: Optional[ArrayLike]
            If provided assigns a node probability score to the objective function.
        """
        if self._nodes is not None:
            raise ValueError("Nodes have already been added.")

        assert_same_length(
            indices=indices,
            is_first_t=is_first_t,
            is_last_t=is_last_t,
            nodes_prob=nodes_prob,
        )

        LOG.info("# %s nodes at starting `t`.", np.sum(is_first_t))
        LOG.info("# %s nodes at last `t`.", np.sum(is_last_t))

        appear_weight = (
            np.logical_not(is_first_t | is_border) * self._config.appear_weight
        )
        disappear_weight = (
            np.logical_not(is_last_t | is_border) * self._config.disappear_weight
        )

        indices = np.asarray(indices, dtype=int)
        self._backward_map = np.array(indices, copy=True)
        self._forward_map = ArrayMap(indices, np.arange(len(indices)))
        size = (len(indices),)

        self._nodes = self._model.add_var_tensor(
            size, name="nodes", var_type=mip.BINARY
        )
        self._appearances = self._model.add_var_tensor(
            size, name="appear", var_type=mip.BINARY
        )
        self._disappearances = self._model.add_var_tensor(
            size, name="disappear", var_type=mip.BINARY
        )
        self._divisions = self._model.add_var_tensor(
            size, name="division", var_type=mip.BINARY
        )

        if nodes_prob is None:
            node_weights = 0
        else:
            nodes_prob = self._config.apply_link_function(np.asarray(nodes_prob))
            node_weights = mip.xsum(nodes_prob * self._nodes)

        self._model.objective = (
            mip.xsum(self._divisions * self._config.division_weight)
            + mip.xsum(self._appearances * appear_weight)
            + mip.xsum(self._disappearances * disappear_weight)
            + node_weights
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
        if self._edges is not None:
            raise ValueError("Edges have already been added.")

        assert_same_length(sources=sources, targets=targets, weights=weights)

        weights = self._config.apply_link_function(weights.astype(float))

        LOG.info("transformed edge weights %s", weights)

        sources = self._forward_map[np.asarray(sources, dtype=int)]
        targets = self._forward_map[np.asarray(targets, dtype=int)]

        self._edges = self._model.add_var_tensor(
            (len(weights),), name="edges", var_type=mip.BINARY
        )
        self._edges_df = pd.DataFrame(
            np.asarray([sources, targets]).T, columns=["sources", "targets"]
        )

        self._model.objective += mip.xsum(weights * self._edges)

    def set_standard_constraints(self) -> None:
        """Sets standard biological/flow constraints:
        - single incoming node (no fusion);
        - flow conservation (begin and end requires slack variables);
        - divisions only from existing nodes;
        """
        edges_targets = self._edges_df.groupby("targets")
        edges_sources = self._edges_df.groupby("sources")

        for i in range(self._nodes.shape[0]):
            # yes, it's flipped, when sources are fixed we access targets
            try:
                i_sources = edges_targets.get_group(i).index
            except KeyError:
                i_sources = []

            try:
                i_targets = edges_sources.get_group(i).index
            except KeyError:
                i_targets = []

            # single incoming node
            self._model.add_constr(
                mip.xsum(self._edges[i_sources]) + self._appearances[i]
                == self._nodes[i]
            )

            # flow conservation
            self._model.add_constr(
                self._nodes[i] + self._divisions[i]
                == mip.xsum(self._edges[i_targets]) + self._disappearances[i]
            )

            # divisions
            self._model.add_constr(self._nodes[i] >= self._divisions[i])

    def add_overlap_constraints(self, sources: ArrayLike, targets: ArrayLike) -> None:
        """Add constraints such that `source` and `target` can't be present in the same solution.

        Parameters
        ----------
        source : ArrayLike
            Source nodes indices.
        target : ArrayLike
            Target nodes indices.
        """
        sources = self._forward_map[np.asarray(sources, dtype=int)]
        targets = self._forward_map[np.asarray(targets, dtype=int)]

        for i in range(len(sources)):
            self._model.add_constr(
                self._nodes[sources[i]] + self._nodes[targets[i]] <= 1
            )

    def enforce_nodes_solution_value(
        self,
        indices: ArrayLike,
        variable: Literal["appear", "disappear", "division", "node"],
        value: bool,
    ) -> None:
        """Constraints given nodes' variables to 1.

        Parameters
        ----------
        indices : ArrayLike
            Nodes indices.
        variable : str
            Slack variable to constraint.
        value : bool
            Value to constraint to.
        """
        indices = self._forward_map[np.asarray(indices, dtype=int)]

        variable_arr = {
            "appear": self._appearances,
            "disappear": self._disappearances,
            "division": self._divisions,
            "node": self._nodes,
        }[variable]

        if value:
            for i in indices:
                self._model.add_constr(variable_arr[i] >= 1)
        else:
            for i in indices:
                self._model.add_constr(variable_arr[i] <= 0)

    def enforce_edges_solution_value(
        self,
        sources: ArrayLike,
        targets: ArrayLike,
        value: bool,
    ) -> None:
        """Constraints given nodes' variables to 1.

        Parameters
        ----------
        source : ArrayLike
            Array of integers indicating source indices (at time T - 1).
        targets : ArrayLike
            Array of integers indicating target indices (at time T).
        value : bool
            Value to constraint to.
        """
        if self._edges_df is None:
            raise ValueError("Edges must be added before enforcing their value.")

        sources = self._forward_map[np.asarray(sources, dtype=int)]
        targets = self._forward_map[np.asarray(targets, dtype=int)]

        # saving indices
        df = self._edges_df.reset_index()

        match = pd.merge(
            df,
            pd.DataFrame({"sources": sources, "targets": targets}),
            on=["sources", "targets"],
            validate="1:1",
        )
        if len(match) != len(sources):
            raise ValueError(
                f"{len(sources) - len(match)} edges were not found at `enforce_edges_solution_value`."
            )

        if value:
            for i in match["index"]:
                self._model.add_constr(self._edges[i] >= 1)
        else:
            for i in match["index"]:
                self._model.add_constr(self._edges[i] <= 0)

    def set_nodes_sum(self, indices: ArrayLike, total_sum: int) -> None:
        """Set indices sum to total_sum as constraint.

        sum_i nodes[i] = total_sum

        Parameters
        ----------
        indices : ArrayLike
            Nodes indices.
        total_sum : int
            Total sum of nodes' variables.
        """
        indices = self._forward_map[np.asarray(indices, dtype=int)]
        self._model.add_constr(mip.xsum([self._nodes[i] for i in indices]) == total_sum)

    def _set_solution_guess(self) -> None:
        # TODO
        pass

    def optimize(self) -> float:
        """Optimizes MIP model."""
        self._model.optimize()
        return self._model.objective_value

    def solution(self) -> pd.DataFrame:
        """Returns the nodes present on the solution.

        Returns
        -------
        pd.DataFrame
            Dataframe indexed by nodes as indices and their parent (NA if orphan).
        """
        if self._model.status == mip.OptimizationStatus.FEASIBLE:
            LOG.warning(
                f"Solver status {self._model.status}. Search interrupted before conclusion."
            )

        elif self._model.status == mip.OptimizationStatus.INFEASIBLE:
            model_path = Path(".") / f"{uuid.uuid4()}_tracking.lp"
            self._model.write(str(model_path.absolute()))
            raise ValueError(
                f"Infeasible solution found. Exported tracking LP to {model_path.absolute()} for debugging."
            )

        elif self._model.status != mip.OptimizationStatus.OPTIMAL:
            raise ValueError(
                f"Solver must be optimized before returning solution. It had status {self._model.status}"
            )

        nodes = np.asarray(
            [i for i, node in enumerate(self._nodes) if node.x > 0.5], dtype=int
        )
        nodes = self._backward_map[nodes]
        LOG.info("Solution nodes\n%s", nodes)

        if len(nodes) == 0:
            raise ValueError("Something went wrong, nodes solution is empty.")

        nodes = pd.DataFrame(
            data=NO_PARENT,
            index=nodes,
            columns=["parent_id"],
        )

        edges_solution = np.asarray(
            [i for i, edge in enumerate(self._edges) if edge.x > 0.5], dtype=int
        )
        edges = self._backward_map[self._edges_df.loc[edges_solution].values]

        LOG.info("Solution edges\n%s", edges)

        if len(edges) == 0:
            raise ValueError("Something went wrong, edges solution is empty")

        edges = pd.DataFrame(
            data=edges[:, 0],
            index=edges[:, 1],
            columns=["parent_id"],
        )

        nodes.update(edges)

        return nodes
