import logging
from typing import Optional, Sequence, Tuple, Union

import zarr
from numpy.typing import ArrayLike
from tqdm import tqdm
from zarr.storage import Store

from ultrack.utils.data import large_chunk_size

LOG = logging.getLogger(__name__)

try:
    import cupy as xp
    import cupyx.scipy.ndimage as ndi
    from cucim.segmentation import find_boundaries

except ImportError:
    LOG.warning("cupy not found, using CPU processing")
    import numpy as xp
    import scipy.ndimage as ndi
    from skimage.segmentation import find_boundaries


def labels_to_edges(
    labels: Union[ArrayLike, Sequence[ArrayLike]],
    sigma: Optional[Union[Sequence[float], float]] = None,
    detection_store: Optional[Store] = None,
    edges_store: Optional[Store] = None,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Converts and merges a sequence of labels into ultrack input format (detection and edges)

    Parameters
    ----------
    labels : Union[ArrayLike, Sequence[ArrayLike]]
        List of labels with equal shape.
    sigma : Optional[Union[Sequence[float], float]], optional
        Edges smoothing parameter (gaussian blur), edges aren't smoothed if not provided
    detection_store : zarr.storage.Store, optional
        Zarr storage, it can be used with zarr.DirectoryStorage to save the output into disk.
        By default it loads the data into memory.
    edges_store : zarr.storage.Store, optional
        Zarr storage, it can be used with zarr.DirectoryStorage to save the output into disk.
        By default it loads the data into memory.
    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        Detection and edges array.
    """

    if detection_store is None:
        detection_store = zarr.MemoryStore()

    if edges_store is None:
        edges_store = zarr.MemoryStore()

    if not isinstance(labels, Sequence):
        labels = [labels]

    shape = labels[0].shape
    for lb in labels:
        if shape != lb.shape:
            raise ValueError(
                f"All labels must have the same shape. Found {shape} and {lb.shape}"
            )

    LOG.info(f"Labels shape {shape}")

    detection = zarr.zeros(
        shape=shape,
        dtype=bool,
        chunks=large_chunk_size(shape, bool),
        store=detection_store,
    )
    edges = zarr.zeros(
        shape=shape,
        dtype=xp.float32,
        chunks=large_chunk_size(shape, xp.float32),
        store=edges_store,
    )

    for t in tqdm(range(shape[0]), "Converting labels to edges"):
        detection_frame = xp.zeros(shape[1:], dtype=detection.dtype)
        edges_frame = xp.zeros(shape[1:], dtype=edges.dtype)

        for lb in labels:
            lb_frame = xp.asarray(lb[t])
            detection_frame |= lb_frame > 0
            edges_frame += find_boundaries(lb_frame, mode="outer")

        edges_frame /= len(labels)

        if sigma is not None:
            edges_frame = ndi.gaussian_filter(edges_frame, sigma)

        if hasattr(edges_frame, "get"):
            # removing from gpu if is cupy array
            edges_frame = edges_frame.get()
            detection_frame = detection_frame.get()

        detection[t] = detection_frame
        edges[t] = edges_frame

    return detection, edges
