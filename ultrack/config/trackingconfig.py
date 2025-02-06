from enum import Enum
from typing import Callable, Literal, Optional, Tuple

import numpy as np
from pydantic.v1 import BaseModel, Extra


class LinkFunctionChoices(Enum):
    identity = "identity"
    power = "power"


class TrackingConfig(BaseModel):
    """Tracking (segmentation & linking selection) configuration"""

    solver_name: Literal["GUROBI", "CBC", ""] = ""
    """
    Constrained optimization solver name.

    * GUROBI: Commercial solver, requires license, see :ref:`gurobi_install` for extra information.

    * CBC: Open-source solver, slower, uses more memory than Gurobi and harder to install on Window.

    * "": Use default solver, GUROBI if available, otherwise CBC.

    """

    appear_weight: float = -0.001
    """Penalization weight for appearing cell, should be negative """
    disappear_weight: float = -0.001
    """Penalization for disappearing cell, should be negative """
    division_weight: float = -0.001
    """Penalization for dividing cell, should be negative """
    image_border_size: Optional[Tuple[int, ...]] = None
    """Image border size in pixels (Z,Y,X) to avoid tracking cells within this border.
       If cells are within the border they not penalized when appearing or disappearing
    """
    n_threads: int = -1
    """Number of worker threads """

    window_size: Optional[int] = None
    """
    Time window size for partially solving the tracking ILP.
    By default it solves the entire timelapse at once.
    Useful for large datasets.
    """

    overlap_size: int = 1
    """
    Number of frames used to shared (overlap/pad) each size when ``window_size`` is set.
    This improves the tracking quality at the edges of the windows and enforce continuity of the tracks.
    """

    solution_gap: float = 0.001
    """
    Solver solution gap. This will speed up the solver when finding the optimal
    solution might taken a long time, but may affect the quality of the solution.
    """

    time_limit: int = 36000
    """Solver execution time limit in seconds """

    method: int = 0
    """``SPECIAL``: Solver method, `reference <https://docs.python-mip.com/en/latest/classes.html#lp-method>`_"""

    link_function: LinkFunctionChoices = "power"
    """``SPECIAL``: Function used to transform the edge and node weights, `identity` or `power`"""

    power: float = 4
    r"""``SPECIAL``: Expoent :math:`\eta` of power transform, :math:`w_{pq}^\eta` """

    bias: float = -0.0
    """``SPECIAL``: Edge weights bias :math:`b`, :math:`w_{pq} + b`, should be negative """

    dismiss_weight_guess: Optional[float] = None
    include_weight_guess: Optional[float] = None

    class Config:
        use_enum_values = True
        extra = Extra.forbid

    @property
    def apply_link_function(self) -> Callable[[np.ndarray], np.ndarray]:
        if self.link_function == "identity":
            return lambda x: x + self.bias
        elif self.link_function == "power":
            return lambda x: np.power(x, self.power) + self.bias
        else:
            raise NotImplementedError
