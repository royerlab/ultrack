import itertools
import logging
import shutil
import warnings
from pathlib import Path
from typing import Callable, Literal, Optional, Tuple, Union

import numpy as np
import sqlalchemy as sqla
import zarr
from numpy.typing import ArrayLike
from sqlalchemy.orm import Session
from tqdm import tqdm
from zarr.storage import Store

from ultrack.core.database import NodeDB

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

    # used during testing, not very useful in practice
    if len(shape) == 2:
        chunks = (1, plane_shape[-1])
    elif len(shape) == 3:
        chunks = (1, *plane_shape)
    else:
        depth = min(max_size // (dtype.itemsize * np.prod(plane_shape)), shape[1])
        chunks = (1,) * (len(shape) - 3) + (depth, *plane_shape)

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

    try:
        in_shape = [arr.shape for arr in in_arrays]
        np.broadcast_shapes(out_array.shape, *in_shape)
    except ValueError as e:
        LOG.warning(
            f"Warning: if you are not using multichannel operations, "
            f"this can be an error. {e}."
        )

    if isinstance(axis, int):
        axis = (axis,)

    stub_slicing = [slice(None) for _ in range(out_array.ndim)]
    multi_indices = list(itertools.product(*[range(out_array.shape[i]) for i in axis]))
    for indices in tqdm(multi_indices, f"Applying {name} ..."):
        for a, i in zip(axis, indices):
            stub_slicing[a] = i
        indexing = tuple(stub_slicing)

        func_result = func(*[a[indexing] for a in in_arrays], **kwargs)
        output_shape = out_array[indexing].shape
        out_array[indexing] = np.broadcast_to(func_result, output_shape)


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


class UltrackArray:
    def __init__(
        self,
        config,
        Tmax,
        database_path: Union[str,None] = None,
        dtype: np.dtype = np.int32,
    ):
        self.config = config
        self.shape = tuple(config.data_config.metadata["shape"])  # (t,(z),y,x)
        self.dtype = dtype
        self.Tmax = Tmax
        self.ndim = len(self.shape)
        self.array = np.zeros(self.shape[1:], dtype=self.dtype)
        self.export_func = self.array.__setitem__

        if database_path is None:
            self.database_path = config.data_config.database_path
        else:
            self.database_path = database_path

        self.minmax = self.find_min_max_volume_entire_dataset()
        self.volume = self.minmax.mean().astype(int)
        
    # proper documentation!!

    def __getitem__(self, indexing):
        if isinstance(indexing, tuple):
            time, volume_slicing = indexing[0], indexing[1:]
        else:
            time = indexing
            volume_slicing = ...

        try:
            time = time.item()  # convert from numpy.int to int
        except:
            time = time

        self.query_volume(
            time=time,
            buffer=self.array,
        )

        return self.array[volume_slicing]

    def query_volume(
        self,
        time: int,
        buffer: np.array,
    ) -> None:
        engine = sqla.create_engine(self.database_path)
        buffer.fill(0)

        with Session(engine) as session:
            query = list(
                session.query(NodeDB.id, NodeDB.pickle, NodeDB.hier_parent_id).where(
                    NodeDB.t == time
                )
            )

            idx_to_plot = []

            for idx, q in enumerate(query):
                if q[1].area <= self.volume:
                    idx_to_plot.append(idx)

            id_to_plot = [q[0] for idx, q in enumerate(query) if idx in idx_to_plot]
            label_list = np.arange(1, len(query) + 1, dtype=int)

            to_remove = []
            for idx in idx_to_plot:
                if query[idx][2] in id_to_plot:  # if parent is also printed
                    to_remove.append(idx)

            for idx in to_remove:
                idx_to_plot.remove(idx)

            if len(query) == 0:
                print("query is empty!")

            for idx in idx_to_plot:
                query[idx][1].paint_buffer(
                    buffer, value=label_list[idx], include_time=False
                )

        return query

    def find_minmax_volumes_1_timepoint(
        self,
        time: int,
    ) -> np.ndarray:

        ##
        # returns an np.array: [minVolume, maxVolume] of all nodes in the hierarchy for a single time point
        ##

        engine = sqla.create_engine(self.database_path)
        min_vol = np.inf
        max_vol = 0
        with Session(engine) as session:
            query = list(session.query(NodeDB.pickle).where(NodeDB.t == time))
            for node in query:
                vol = node[0].area
                if vol < min_vol:
                    min_vol = vol
                if vol > max_vol:
                    max_vol = vol
        return np.array([min_vol, max_vol]).astype(int)

    def find_min_max_volume_entire_dataset(self):
        ##
        # loops over all time points in the stack and returns an
        # np.array: [minVolume, maxVolume] of all nodes in the hierarchy over all times
        ##
        min_vol = np.inf
        max_vol = 0
        for t in range(self.Tmax): #range(self.shape[0]):
            minmax = self.find_minmax_volumes_1_timepoint(t)
            if minmax[0] < min_vol:
                min_vol = minmax[0]
            if minmax[1] > max_vol:
                max_vol = minmax[1]

        return np.array([min_vol, max_vol], dtype=int)
