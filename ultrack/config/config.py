import logging
from pathlib import Path
from typing import Optional, Union

import toml
from pydantic.v1 import BaseModel, Extra, Field

from ultrack.config.dataconfig import DataConfig
from ultrack.config.segmentationconfig import SegmentationConfig
from ultrack.config.trackingconfig import TrackingConfig

LOG = logging.getLogger(__name__)


class LinkingConfig(BaseModel):
    """
    Candidate cell hypotheses linking configuration
    """

    max_distance: float = 15.0
    """Maximum distance between neighboring segments """

    n_workers: int = 1
    """Number of worker threads """

    max_neighbors: int = 5
    """Maximum number of neighbors per candidate segment """

    distance_weight: float = 0.0
    r"""
    Penalization weight :math:`\gamma` for distance between segment centroids,
    :math:`w_{pq} - \gamma \|c_p - c_q\|_2`, where :math:`c_p` is region :math:`p` center of mass
    """

    z_score_threshold: float = 5.0
    """
    ``SPECIAL``: z-score threshold between intensity values from within
    the segmentation masks of neighboring segments
    """

    class Config:
        extra = Extra.forbid


class MainConfig(BaseModel):
    data_config: Optional[DataConfig] = Field(
        default_factory=DataConfig, alias="data", nullable=True
    )
    """
    Configuration for intermediate data storage and retrieval.
    """

    segmentation_config: SegmentationConfig = Field(
        default_factory=SegmentationConfig, alias="segmentation"
    )
    """Segmentation hypotheses creation configuration """

    linking_config: LinkingConfig = Field(
        default_factory=LinkingConfig, alias="linking"
    )
    """Candidate cell hypotheses linking configuration"""

    tracking_config: TrackingConfig = Field(
        default_factory=TrackingConfig, alias="tracking"
    )
    """Tracking (segmentation & linking selection) configuration """


def load_config(path: Union[str, Path]) -> MainConfig:
    """Creates MainConfig from TOML file."""
    with open(path) as f:
        data = toml.load(f)
        LOG.info(data)
        return MainConfig.parse_obj(data)
