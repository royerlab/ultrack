from abc import ABC, abstractmethod
from typing import Literal

import pandas as pd
from numpy.typing import ArrayLike

from ultrack.config.config import TrackingConfig


class BaseSolver(ABC):
    def __init__(
        self,
        config: TrackingConfig,
    ) -> None:
        """
        Abstract base solver for api specification.

        Parameters
        ----------
        config : TrackingConfig
            Tracking configuration parameters.
        """
        self._config = config

    @staticmethod
    def _assert_same_length(**kwargs) -> None:
        """Validates if key-word arguments have the same length."""
        for k1, v1 in kwargs.items():
            for k2, v2 in kwargs.items():
                if len(v2) != len(v1):
                    raise ValueError(
                        f"`{k1}` and `{k2}` must have the same length. Found {len(v1)} and {len(v2)}."
                    )

    @abstractmethod
    def add_nodes(
        self, indices: ArrayLike, is_first_t: ArrayLike, is_last_t: ArrayLike
    ) -> None:
        """Add nodes variables solver.

        Parameters
        ----------
        indices : ArrayLike
            Nodes indices.
        is_first_t : ArrayLike
            Boolean array indicating if it belongs to first time point and it won't receive appearance penalization.
        is_last_t : ArrayLike
            Boolean array indicating if it belongs to last time point and it won't receive disappearance penalization.
        """

    @abstractmethod
    def add_edges(
        self, sources: ArrayLike, targets: ArrayLike, weights: ArrayLike
    ) -> None:
        """Add edges variables to solver.

        Parameters
        ----------
        source : ArrayLike
            Array of integers indicating source indices.
        targets : ArrayLike
            Array of integers indicating target indices.
        weights : ArrayLike
            Array of weights, input to the link function.
        """

    def set_standard_constraints(self) -> None:
        """Sets standard biological/flow constraints:
        - single incoming node (no fusion);
        - flow conservation (begin and end requires slack variables);
        - divisions only from existing nodes;
        """

    @abstractmethod
    def add_overlap_constraints(self, sources: ArrayLike, targets: ArrayLike) -> None:
        """Add constraints such that `source` and `target` can't be present in the same solution.

        Parameters
        ----------
        sources : ArrayLike
            Source nodes indices.
        targets : ArrayLike
            Target nodes indices.
        """

    @abstractmethod
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

    @abstractmethod
    def optimize(self) -> float:
        """
        Computes solution.

        Returns
        -------
        float
            Solution objective value.
        """

    @abstractmethod
    def solution(self) -> pd.DataFrame:
        """Returns the nodes present on the solution.

        Returns
        -------
        pd.DataFrame
            Dataframe indexed by nodes as indices and their parent (NA if orphan).
        """
