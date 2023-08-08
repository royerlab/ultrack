import logging
from pathlib import Path
from typing import Union

import toml
from pydantic import BaseModel, Extra, Field

from ultrack.config.dataconfig import DataConfig
from ultrack.config.segmentationconfig import SegmentationConfig
from ultrack.config.trackingconfig import TrackingConfig

LOG = logging.getLogger(__name__)


class LinkingConfig(BaseModel):
    n_workers: int = 1
    max_neighbors: int = 5
    max_distance: float = 15.0
    distance_weight: float = 0.0
    z_score_threshold: float = 5.0

    class Config:
        extra = Extra.forbid


class MainConfig(BaseModel):
    data_config: DataConfig = Field(default_factory=DataConfig, alias="data")
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
