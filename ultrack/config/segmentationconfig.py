from typing import Any, Callable, Dict, Literal

import higra as hg
from pydantic.v1 import BaseModel, Extra, root_validator, validator

NAME_TO_WS_HIER = {
    "area": hg.watershed_hierarchy_by_area,
    "dynamics": hg.watershed_hierarchy_by_dynamics,
    "volume": hg.watershed_hierarchy_by_volume,
}


class SegmentationConfig(BaseModel):
    """Segmentation hypotheses creation configuration"""

    min_area: int = 100
    """
    Minimum segment number of pixels, regions smaller than this value are merged
    or removed when there is no neighboring region
    """

    max_area: int = 1_000_000
    """Maximum segment's number of pixels, regions larger than this value are removed """

    n_workers: int = 1
    """Number of worker threads """

    min_frontier: float = 0.0
    """
    Minimum average frontier value between candidate segmentations, regions sharing an average
    frontier value lower than this are merged
    """

    threshold: float = 0.5
    """Threshold used to binarize the cell foreground map"""

    max_noise: float = 0.0
    """``SPECIAL``: Upper limit of uniform distribution for additive noise on contour map """

    random_seed: Literal["frame", "none", None] = "frame"
    """``SPECIAL``: Random seed initialization, if `frame` the seed is the timelapse frame number """

    ws_hierarchy: Callable = hg.watershed_hierarchy_by_area
    """
    ``SPECIAL``: Watershed hierarchy function from
    `higra <https://higra.readthedocs.io/en/stable/python/watershed_hierarchy.html>`_
    used to construct the hierarchy
    """

    anisotropy_penalization: float = 0.0
    """
    ``SPECIAL``: Image graph z-axis penalization, positive values will prioritize segmenting
    the xy-plane first, negative will do the opposite
    """

    class Config:
        use_enum_values = True
        extra = Extra.forbid

    @validator("ws_hierarchy", pre=True)
    def ws_name_to_function(cls, value: str) -> Callable:
        """Converts string to watershed hierarchy function."""
        if not isinstance(value, str):
            raise ValueError(
                f"`ws_hierarchy` must be a string. Found {value} of type {type(value)}."
            )

        if value not in NAME_TO_WS_HIER:
            raise ValueError(
                f"`ws_hierarchy` must match {NAME_TO_WS_HIER.keys()}. Found {value}"
            )

        return NAME_TO_WS_HIER[value]

    @root_validator
    def area_validator(cls, values: Dict) -> Dict:
        """Checks if min and max area are compatible."""
        if values["min_area"] > values["max_area"]:
            raise ValueError(
                "`min_area` must be lower than `max_area`."
                f"Found min_area={values['min_area']}, max_area={values['max_area']}"
            )
        return values

    def dict(self, *args, **kwargs) -> Dict[str, Any]:
        d = super().dict(*args, **kwargs)
        for name, func in NAME_TO_WS_HIER.items():
            if func == d["ws_hierarchy"]:
                d["ws_hierarchy"] = name
                break
        return d
