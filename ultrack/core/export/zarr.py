from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
import zarr
from zarr.storage import Store

from ultrack.config.config import MainConfig
from ultrack.core.export.utils import export_segmentation_generic
from ultrack.utils.array import create_zarr, large_chunk_size


def tracks_to_zarr(
    config: MainConfig,
    tracks_df: pd.DataFrame,
    store_or_path: Union[None, Store, Path, str] = None,
    chunks: Optional[Tuple[int]] = None,
    overwrite: bool = False,
) -> zarr.Array:
    """
    Exports segmentations masks to zarr array, `track_df` assign the `track_id` to their respective segments.
    By changing the `store` this function can be used to write zarr arrays into disk.

    Parameters
    ----------
    config : MainConfig
        Configuration parameters.
    tracks_df : pd.DataFrame
        Tracks dataframe, must have `track_id` column and be indexed by node id.
    store_or_path : Union[None, Store, Path, str], optional
        Zarr storage or output path, if not provided zarr.TempStore is used.
    chunks : Optional[Tuple[int]], optional
        Chunk size, if not provided it chunks time with 1 and the spatial dimensions as big as possible.
    overwrite : bool, optional
        If True, overwrites existing zarr array.

    Returns
    -------
    zarr.Array
        Output zarr array.
    """

    shape = config.data_config.metadata["shape"]
    dtype = np.int32

    if isinstance(store_or_path, zarr.MemoryStore) and config.data_config.n_workers > 1:
        raise ValueError(
            "zarr.MemoryStore and multiple workers are not allowed. "
            f"Found {config.data_config.n_workers} workers in `data_config`."
        )

    if chunks is None:
        chunks = large_chunk_size(shape, dtype=dtype)

    if isinstance(store_or_path, Store):
        array = zarr.zeros(shape, dtype=dtype, store=store_or_path, chunks=chunks)

    else:
        array = create_zarr(
            shape,
            dtype=dtype,
            store_or_path=store_or_path,
            chunks=chunks,
            default_store_type=zarr.TempStore,
            overwrite=overwrite,
        )

    export_segmentation_generic(config.data_config, tracks_df, array.__setitem__)
    return array
