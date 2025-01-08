import os
import warnings

if os.environ.get("ULTRACK_DEBUG", False):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

# ignoring small float32/64 zero flush warning
warnings.filterwarnings("ignore", message="The value of the smallest subnormal for")

__version__ = "0.6.3"

from ultrack.config.config import MainConfig, load_config
from ultrack.core.export.ctc import to_ctc
from ultrack.core.export.exporter import export_tracks_by_extension
from ultrack.core.export.trackmate import to_trackmate
from ultrack.core.export.tracks_layer import to_tracks_layer
from ultrack.core.export.zarr import tracks_to_zarr
from ultrack.core.interactive import add_new_node
from ultrack.core.linking.processing import link
from ultrack.core.main import track
from ultrack.core.segmentation.processing import segment
from ultrack.core.solve.processing import solve
from ultrack.core.tracker import Tracker
from ultrack.imgproc.flow import add_flow
