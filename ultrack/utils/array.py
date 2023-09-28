import itertools
import logging
import shutil
import warnings
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
import zarr
from numpy.typing import ArrayLike
from tqdm import tqdm
from zarr.storage import Store

LOG = logging.getLogger(__name__)


def large_chunk_size(
    shape: Tuple[int],
    dtype: Union[str, np.dtype],
    max_size: int = 2147483647,
) -> Tuple[int]:
    """
    Computes a large chunk size for a given `shape` and `dtype`.
    Large chunks improves the performance on Elastic Storage Systems (ESS).
    Leading dimension (time) will always be chunked as 1.

    Parameters
    ----------
    shape : Tuple[int]
        Input data shape.
    dtype : Union[str, np.dtype]
        Input data type.
    max_size : int, optional
        Reference maximum size, by default 2147483647

    Returns
    -------
    Tuple[int]
        Suggested chunk size.
    """
    if not isinstance(dtype, np.dtype):
        dtype = np.dtype(dtype)

    plane_shape = np.minimum(shape[-2:], 32768)

    if len(shape) == 3:
        chunks = (1, *plane_shape)
    elif len(shape) > 3:
        depth = min(max_size // (dtype.itemsize * np.prod(plane_shape)), shape[1])
        chunks = (1,) * (len(shape) - 3) + (depth, *plane_shape)
    else:
        raise NotImplementedError(
            f"Large chunk size only implemented for 3-or-more dimensional arrays. Found {len(shape) - 1}-dims."
        )

    return chunks


def validate_and_overwrite_path(
    path: Path, overwrite: bool, msg_type: Literal["cli", "api"]
) -> None:
    """Validates and errors existance of path (or dir) and overwrites it if requested."""

    if msg_type == "cli":
        msg = f"{path} already exists. Set `--overwrite` option to overwrite it."

    elif msg_type == "api":
        msg = f"{path} already exists. Set `overwrite=True` to overwrite it."

    else:
        raise ValueError(f"Invalid `msg_type` {msg_type}, must be `cli` or `api`.")

    if path.exists():
        if overwrite:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        else:
            raise ValueError(msg)


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


def array_apply(
    *in_arrays: ArrayLike,
    out_array: ArrayLike,
    func: Callable,
    axis: Union[Tuple[int], int] = 0,
    **kwargs,
) -> None:
    """Apply a function over a given dimension of an array.

    Parameters
    ----------
    in_arrays : ArrayLike
        Arrays to apply function to.
    out_array : ArrayLike
        Array to store result of function.
    func : function
        Function to apply over time.
    axis : Union[Tuple[int], int], optional
        Axis of data to apply func, by default 0.
    args : tuple
        Positional arguments to pass to func.
    **kwargs :
        Keyword arguments to pass to func.
    """
    name = func.__name__ if hasattr(func, "__name__") else type(func).__name__

    for arr in in_arrays:
        if arr.shape != out_array.shape:
            raise ValueError(
                f"Input arrays {arr.shape} must have the same shape as the output array {out_array.shape}."
            )

    if isinstance(axis, int):
        axis = (axis,)

    stub_slicing = [slice(None) for _ in range(out_array.ndim)]
    multi_indices = list(itertools.product(*[range(out_array.shape[i]) for i in axis]))
    for indices in tqdm(multi_indices, f"Applying {name} ..."):
        for a, i in zip(axis, indices):
            stub_slicing[a] = i
        indexing = tuple(stub_slicing)
        out_array[indexing] = func(*[a[indexing] for a in in_arrays], **kwargs)


def create_zarr(
    shape: Tuple[int, ...],
    dtype: np.dtype,
    store_or_path: Union[Store, Path, str, None] = None,
    overwrite: bool = False,
    default_store_type: Store = zarr.MemoryStore,
    chunks: Optional[Tuple[int]] = None,
    **kwargs,
) -> zarr.Array:
    """Create a zarr array of zeros.

    Parameters
    ----------
    shape : Tuple[int, ...]
        Shape of the array.
    dtype : np.dtype
        Data type of the array.
    store_or_path : Optional[Union[Path, str]], optional
        Path to store the array, if None a zarr.MemoryStore is used, by default None
    overwrite : bool, optional
        Overwrite existing file, by default False
    chunks : Optional[Tuple[int]], optional
        Chunk size, if not provided it chunks time with 1 and the spatial dimensions as big as possible.

    Returns
    -------
    zarr.Array
        Zarr array of zeros.
    """
    if "path" in kwargs:
        raise ValueError("`path` is not a valid argument, use `store_or_path` instead.")

    if store_or_path is None:
        store = default_store_type()

    elif isinstance(store_or_path, Store):
        store = store_or_path

    else:
        if isinstance(store_or_path, str):
            store_or_path = Path(store_or_path)

        validate_and_overwrite_path(store_or_path, overwrite, msg_type="api")

        store = zarr.NestedDirectoryStore(str(store_or_path))

    if chunks is None:
        chunks = large_chunk_size(shape, dtype=dtype)

    return zarr.zeros(shape, dtype=dtype, store=store, chunks=chunks, **kwargs)
