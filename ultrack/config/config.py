import logging
from enum import Enum
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


LOG = logging.getLogger(__name__)


class DataBaseChoices(Enum):
    sqlite = "sqlite"


class DataConfig(BaseModel):
    working_dir: Path = Path(".")
    database: DataBaseChoices = "sqlite"

    @validator("working_dir")
    def validate_working_dir_writeable(cls, value: Path) -> Path:
        """Converts string to watershed hierarchy function."""

        value.mkdir(exist_ok=True)
        try:
            tmp_path = value / ".write_test"
            file_handle = open(tmp_path, "w")
            file_handle.close()
            tmp_path.unlink()
        except OSError:
            ValidationError(f"Working directory {value} isn't writable.")

        return value

    @property
    def database_path(self) -> str:
        """Returns database path given working directory and database type."""
        if self.database == "sqlite":
            return f"sqlite:///{self.working_dir.absolute()}/data.db"
        else:
            raise NotImplementedError(
                f"Dataset type {self.database} support not implemented."
            )


class ReaderConfig(BaseModel):
    reader_plugin: str = "builtins"
    layer_indices: List[Union[str, int]] = [0, 1]

    @validator("layer_indices", pre=True)
    def validate_layer_index_length(cls, value: List) -> List:
        """Checks if layer_index has length 2"""
        if len(value) != 2:
            ValidationError(f"`layer_indices` must have length 2. Found {len(value)}.")

        return value


class SegmentationConfig(BaseModel):
    threshold: float = 0.5
    min_area: int = 100
    max_area: int = 1_000_000
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


class LinkingConfig(BaseModel):
    n_workers: int = 1
    max_neighbors: int = 10
    max_distance: float = 15.0


class TrackingConfig(BaseModel):
    appear_weight: float = -0.5
    disappear_weight: float = -0.75
    division_weight: float = -1.0
    dismiss_weight_guess: Optional[float] = None
    include_weight_guess: Optional[float] = None
    solution_gap: float = 0.001
    time_limit: int = 36000
    method: int = -1
    n_threads: int = -1
    edge_transform: Callable = None  # FIXME: implement identity


class MainConfig(BaseModel):
    data_config: DataConfig = Field(default_factory=DataConfig, alias="data")
    reader_config: ReaderConfig = Field(default_factory=ReaderConfig, alias="reader")
    segmentation_config: SegmentationConfig = Field(
        default_factory=SegmentationConfig, alias="segmentation"
    )
    linking_config: LinkingConfig = Field(
        default_factory=LinkingConfig, alias="linking"
    )
    tracking_config: TrackingConfig = Field(
        default_factory=TrackingConfig, alias="tracking"
    )


def load_config(path: Union[str, Path]) -> MainConfig:
    """Creates MainConfig from TOML file."""
    with open(path) as f:
        data = toml.load(f)
        LOG.info(data)
        return MainConfig.parse_obj(data)
