from enum import Enum
from typing import Callable, Optional

import numpy as np
from pydantic import BaseModel, Extra


class LinkFunctionChoices(Enum):
    identity = "identity"
    power = "power"


class TrackingConfig(BaseModel):
    appear_weight: float = -0.001
    disappear_weight: float = -0.001
    division_weight: float = -0.001
    dismiss_weight_guess: Optional[float] = None
    include_weight_guess: Optional[float] = None
    window_size: Optional[int] = None
    overlap_size: int = 1
    solution_gap: float = 0.001
    time_limit: int = 36000
    method: int = 0
    n_threads: int = -1
    link_function: LinkFunctionChoices = "power"
    power: float = 4
    bias: float = -0.0

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
