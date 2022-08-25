from typing import Any, Callable, Dict

import higra as hg
from pydantic import BaseModel, ValidationError, validator

NAME_TO_WS_HIER = {
    "area": hg.watershed_hierarchy_by_area,
    "dynamics": hg.watershed_hierarchy_by_dynamics,
    "volume": hg.watershed_hierarchy_by_volume,
}


class SegmentationConfig(BaseModel):
    threshold: float = 0.5
    min_area: int = 100
    max_area: int = 1_000_000
    min_frontier: float = 0.0
    anisotropy_penalization: float = 0.0
    ws_hierarchy: Callable = hg.watershed_hierarchy_by_area
    n_workers: int = 1

    @validator("ws_hierarchy", pre=True)
    def ws_name_to_function(cls, value: str) -> Callable:
        """Converts string to watershed hierarchy function."""
        if not isinstance(value, str):
            ValidationError(
                f"`ws_hierarchy` must be a string. Found {value} of type {type(value)}."
            )

        if value not in NAME_TO_WS_HIER:
            ValidationError(
                f"`ws_hierarchy` must match {NAME_TO_WS_HIER.keys()}. Found {value}"
            )

        return NAME_TO_WS_HIER[value]

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        d = super().dict(*args, **kwargs)
        for name, func in NAME_TO_WS_HIER.items():
            if func == d["ws_hierarchy"]:
                d["ws_hierarchy"] = name
                break
        return d
