from pathlib import Path
from typing import Callable, List, Optional, Union

import higra as hg
import toml
from pydantic import BaseModel, Field, ValidationError, validator

NAME_TO_WS_HIER = {
    "area": hg.watershed_hierarchy_by_area,
    "dynamics": hg.watershed_hierarchy_by_dynamics,
    "volume": hg.watershed_hierarchy_by_volume,
}


class ReaderConfig(BaseModel):
    reader_plugin: str = "builtins"
    layer_names: List[Union[str, int]] = [0, 1]


class InitConfig(BaseModel):
    threshold: float
    max_area: int
    min_area: int
    min_frontier: float = 0.0
    anisotropy_penalization: float = 0.0
    ws_hierarchy: Callable = hg.watershed_hierarchy_by_area
    n_workers: int = 1
    max_neighbors: int = 10
    max_distance: float = 15.0

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


class ComputeConfig(BaseModel):
    appear_weight: float
    disappear_weight: float
    division_weight: float
    dismiss_weight_guess: Optional[float] = None
    include_weight_guess: Optional[float] = None
    solution_gap: float = 0.001
    time_limit: int = 36000
    method: int = -1
    n_threads: int = -1
    edge_transform: Callable = None  # FIXME: implement identity


class MainConfig(BaseModel):
    working_dir: Path = Path(".")
    reader_config: ReaderConfig = Field(alias="reader")
    init_config: InitConfig = Field(alias="init")
    compute_config: ComputeConfig = Field(alias="compute")


def load_config(path: Union[str, Path]) -> MainConfig:
    """Creates MainConfig from TOML file."""
    with open(path) as f:
        return MainConfig.parse_obj(toml.load(f))
