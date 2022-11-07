from typing import Optional, Tuple

import numpy as np
import pandas as pd
import zarr
from zarr.storage import Store

from ultrack.config.dataconfig import DataConfig
from ultrack.core.export.utils import export_segmentation_generic
from ultrack.utils import large_chunk_size


def tracks_to_zarr(
    data_config: DataConfig,
    tracks_df: pd.DataFrame,
    store: Optional[Store] = None,
    chunks: Optional[Tuple[int]] = None,
) -> zarr.Array:
    """
    Exports segmentations masks to zarr array, `track_df` assign the `track_id` to their respective segments.
    By changing the `store` this function can be used to write zarr arrays into disk.

    Parameters
    ----------
    data_config : DataConfig
        Data configuration parameters.
    tracks_df : pd.DataFrame
        Tracks dataframe, must have `track_id` column and be indexed by node id.
    store : Optional[Store], optional
        Zarr storage, if not provided zarr.MemoryStore is used.
    chunks : Optional[Tuple[int]], optional
        Chunk size, if not provided it chunks time with 1 and the spatial dimensions as big as possible.

    Returns
    -------
    zarr.Array
        Output zarr array.
    """

    shape = data_config.metadata["shape"]

    if store is None:
        store = zarr.MemoryStore()

    if chunks is None:
        chunks = large_chunk_size(shape, dtype=np.uint16)

    array = zarr.zeros(shape, dtype=np.uint16, store=store, chunks=chunks)
    export_segmentation_generic(
        data_config, tracks_df, lambda t, buffer: array.__setitem__(t, buffer)
    )
    return array
