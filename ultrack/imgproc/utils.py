from typing import List, Optional

import numpy as np
from numpy.typing import ArrayLike


def _parse_voxel_size(
    voxel_size: Optional[ArrayLike],
    ndim: int,
    channel_axis: Optional[int],
) -> np.ndarray:
    """Parses voxel size and returns it in the correct format."""
    ndim = ndim - (channel_axis is not None)

    if voxel_size is None:
        voxel_size = (1.0,) * ndim

    if len(voxel_size) != ndim:
        raise ValueError(
            "Voxel size must have the same number of elements as the image dimensions."
            f"Expected {ndim} got {len(voxel_size)}."
        )

    return np.asarray(voxel_size)


def _channel_iterator(
    image: ArrayLike,
    channel_axis: Optional[int],
) -> List[Optional[int]]:
    """Iterates over channels, it returns a [None] if channel_axis is None."""

    if channel_axis is None:
        channel_iterator = [None]
    else:
        channel_iterator = range(image.shape[channel_axis])

    return channel_iterator
