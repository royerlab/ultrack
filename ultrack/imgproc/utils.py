import shutil
from pathlib import Path
from typing import Callable, Optional, Tuple, Union

import numpy as np
import zarr
from numpy.typing import ArrayLike
from tqdm import tqdm
from zarr.storage import Store

from ultrack.utils import large_chunk_size


def array_apply(
    func: Callable,
    in_data: ArrayLike,
    out_data: ArrayLike,
    *,
    axis: int = 0,
    **kwargs,
) -> None:
    """Apply a function over a given dimension of an array.

    Parameters
    ----------
    func : function
        Function to apply over time.
    in_data : zarr.Array
        Zarr array to apply function to.
    out_data : zarr.Array
        Zarr array to store result of function.
    axis : int, optional
        Axis of data to apply func, by default 0.
    args : tuple
        Positional arguments to pass to func.
    **kwargs :
        Keyword arguments to pass to func.
    """
    stub_slicing = (slice(None),) * axis
    for i in tqdm(range(in_data.shape[axis]), f"Applying {func.__name__} ..."):
        indexing = stub_slicing + (i,)
        out_data[indexing] = func(in_data[indexing], **kwargs)


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
    path : Optional[Union[Path, str]], optional
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
    if store_or_path is not None:
        if isinstance(store_or_path, str):
            store_or_path = Path(store_or_path)

        if store_or_path.exists():
            if not overwrite:
                raise FileExistsError(f"File {store_or_path} already exists.")
            else:
                shutil.rmtree(store_or_path)

        store = zarr.DirectoryStore(str(store_or_path))
    else:
        store = default_store_type()

    if chunks is None:
        chunks = large_chunk_size(shape, dtype=dtype)

    return zarr.zeros(shape, dtype=dtype, store=store, chunks=chunks, **kwargs)
