import warnings

from numpy.typing import ArrayLike


def check_array_chunk(array: ArrayLike) -> None:
    """Checks if chunked array has chunk size of 1 on time dimension."""
    if hasattr(array, "chunks"):
        if array.chunks[0] != 1:
            warnings.warn(
                f"Array not chunked over time dimension. Found chunk of shape {array.chunk}."
                "Performance will be slower."
            )
