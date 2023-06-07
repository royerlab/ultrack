import logging
import warnings

from numpy.typing import ArrayLike

LOG = logging.getLogger(__name__)


def check_array_chunk(array: ArrayLike) -> None:
    """Checks if chunked array has chunk size of 1 on time dimension."""
    if hasattr(array, "chunks"):
        chunk_shape = array.chunks
        if isinstance(chunk_shape[0], tuple):
            # sometimes the chunk shapes items are tuple, I don't know why
            chunk_shape = chunk_shape[0]
        if chunk_shape[0] != 1:
            warnings.warn(
                f"Array not chunked over time dimension. Found chunk of shape {array.chunks}."
                "Performance will be slower."
            )
