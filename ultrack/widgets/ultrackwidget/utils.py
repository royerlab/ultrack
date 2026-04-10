from enum import Enum

from napari.layers import Layer
from napari.types import ArrayLike


class UltrackInput(Enum):
    IMAGE = "image"
    CONTOURS = "contours"
    DETECTION = "detection"
    LABELS = "labels"


def get_layer_data(layer: Layer) -> ArrayLike:
    """Return the array data from a napari layer.

    For multi-scale layers, returns the highest-resolution level (index 0).

    Parameters
    ----------
    layer : Layer
        A napari layer, possibly multi-scale.

    Returns
    -------
    ArrayLike
        The underlying array data.
    """
    if getattr(layer, "multiscale", False):
        return layer.data[0]
    return layer.data
