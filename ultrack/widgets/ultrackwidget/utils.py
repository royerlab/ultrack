from enum import Enum

from ultrack.utils.napari import get_layer_data  # noqa: F401


class UltrackInput(Enum):
    IMAGE = "image"
    CONTOURS = "contours"
    DETECTION = "detection"
    LABELS = "labels"
