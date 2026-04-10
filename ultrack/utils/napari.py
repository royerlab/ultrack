from napari.layers import Layer
from napari.types import ArrayLike


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
    if layer.multiscale:
        return layer.data[0]
    return layer.data
