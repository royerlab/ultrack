import logging
from functools import wraps
from typing import Optional, Sequence, Tuple, Union

from numpy.typing import ArrayLike
from tqdm import tqdm
from zarr.storage import StoreLike

from ultrack.utils.array import create_zarr
from ultrack.utils.cuda import import_module, to_cpu, xp
from ultrack.utils.deprecation import rename_argument

LOG = logging.getLogger(__name__)


@rename_argument("detection_store_or_path", "foreground_store_or_path")
@rename_argument("edges_store_or_path", "contours_store_or_path")
def labels_to_contours(
    labels: Union[ArrayLike, Sequence[ArrayLike]],
    sigma: Optional[Union[Sequence[float], float]] = None,
    foreground_store_or_path: Optional[StoreLike] = None,
    contours_store_or_path: Optional[StoreLike] = None,
    overwrite: bool = False,
) -> Tuple[ArrayLike, ArrayLike]:
    """
    Converts and merges a sequence of labels into ultrack input format (foreground and contours)

    Parameters
    ----------
    labels : Union[ArrayLike, Sequence[ArrayLike]]
        List of labels with equal shape.
    sigma : Optional[Union[Sequence[float], float]], optional
        Contours smoothing parameter (gaussian blur), contours aren't smoothed when not provided.
    foreground_store_or_path : str, zarr.storage.Store, optional
        Zarr storage, it can be used with zarr.storage.LocalStore to save the output into disk.
        By default it creates a temporary store.
    contours_store_or_path : str, zarr.storage.Store, optional
        Zarr storage, it can be used with zarr.storage.LocalStore to save the output into disk.
        By default it creates a temporary store.
    overwrite : bool, optional
        Overwrite output output files if they already exist, by default False.

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        Combined foreground and edges arrays.
    """
    ndi = import_module("scipy", "ndimage")
    segm = import_module("skimage", "segmentation")

    if not isinstance(labels, Sequence):
        labels = [labels]

    shape = labels[0].shape
    for lb in labels:
        if shape != lb.shape:
            raise ValueError(
                f"All labels must have the same shape. Found {shape} and {lb.shape}"
            )

    LOG.info(f"Labels shape {shape}")

    foreground = create_zarr(
        shape=shape,
        dtype=bool,
        store_or_path=foreground_store_or_path,
        overwrite=overwrite,
        default_store_type=None,
    )
    contours = create_zarr(
        shape=shape,
        dtype=xp.float32,
        store_or_path=contours_store_or_path,
        overwrite=overwrite,
        default_store_type=None,
    )

    for t in tqdm(range(shape[0]), "Converting labels to contours"):
        foreground_frame = xp.zeros(shape[1:], dtype=foreground.dtype)
        contours_frames = xp.zeros(shape[1:], dtype=contours.dtype)

        for lb in labels:
            lb_frame = xp.asarray(lb[t])
            foreground_frame |= lb_frame > 0
            contours_frames += segm.find_boundaries(lb_frame, mode="outer")

        contours_frames /= len(labels)

        if sigma is not None:
            contours_frames = ndi.gaussian_filter(contours_frames, sigma)
            contours_frames = contours_frames / contours_frames.max()

        foreground[t] = to_cpu(foreground_frame)
        contours[t] = to_cpu(contours_frames)

    return foreground, contours


@wraps(labels_to_contours)
def labels_to_edges(*args, **kwargs) -> Tuple[ArrayLike, ArrayLike]:
    return labels_to_contours(*args, **kwargs)
