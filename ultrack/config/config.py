import logging
from pathlib import Path
from typing import List, Union

import toml
from pydantic import BaseModel, Field, ValidationError, validator

from ultrack.config.dataconfig import DataConfig
from ultrack.config.segmentationconfig import SegmentationConfig
from ultrack.config.trackingconfig import TrackingConfig

LOG = logging.getLogger(__name__)


class ReaderConfig(BaseModel):
    reader_plugin: str = "builtins"
    layer_indices: List[Union[str, int]] = [0, 1]

    @validator("layer_indices", pre=True)
    def validate_layer_index_length(cls, value: List) -> List:
        """Checks if layer_index has length 2"""
        if len(value) != 2:
            ValidationError(f"`layer_indices` must have length 2. Found {len(value)}.")

        return value


class LinkingConfig(BaseModel):
    n_workers: int = 1
    max_neighbors: int = 10
    max_distance: float = 15.0


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
