import logging
from typing import Callable, Optional, Union

import numpy as np
import zarr
from numpy.typing import ArrayLike
from tqdm import tqdm
from zarr.storage import Store

from ultrack.utils.array import create_zarr
from ultrack.utils.cuda import import_module, to_cpu

LOG = logging.getLogger(__name__)


def register_timelapse(
    timelapse: ArrayLike,
    *,
    store_or_path: Union[Store, str, None] = None,
    overwrite: bool = False,
    to_device: Callable[[ArrayLike], ArrayLike] = lambda x: x,
    reference_channel: Optional[int] = None,
    overlap_ratio: float = 0.25,
    normalization: Optional[str] = None,
    padding: Optional[int] = None,
    **kwargs,
) -> zarr.Array:
    """
    Register a timelapse sequence using phase cross correlation.

    Parameters
    ----------
    timelapse : ArrayLike
        Input timelapse sequence, T(CZ)YX array C and Z are optional.
        NOTE: when provided, C must be the second dimension after T.
    store_or_path : Union[Store, str, None], optional
        Zarr storage or a file path, to save the output, useful for larger than memory datasets.
        By default it loads the data into memory.
    overwrite : bool, optional
        Overwrite output file if it already exists, when using directory store or a path.
    to_device : Callable[[ArrayLike], ArrayLike], optional
        Function to move the input data to cuda device, by default lambda x: x (CPU).
    reference_channel : Optional[int], optional
        Reference channel for registration, by default None.
        It must be provided when it contains a channel dimension.
    overlap_ratio : float, optional
        Overlap ratio for phase cross correlation, by default 0.25.
    normalization : Optional[str], optional
        Normalization method for phase cross correlation, by default None.
    padding : Optional[int], optional
        Padding for registration, by default None.
    **kwargs
        Additional arguments for phase cross correlation. See `skimage.registration phase_cross_correlation
        <https://scikit-image.org/docs/stable/api/skimage.registration.html#skimage.registration.phase_cross_correlation>`_.  # noqa: E501

    Returns
    -------
    zarr.Array
        Registered timelapse sequence.
    """
    shape = list(timelapse.shape)

    if padding is not None:
        offset = 1 if reference_channel is None else 2
        pads = [(0, 0)] * (offset - 1)

        for i in range(offset, len(shape)):
            shape[i] += 2 * padding
            pads.append((padding, padding))

        def maybe_pad(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            x = to_device(x)
            return np.pad(x, pads, mode="constant")

    else:

        def maybe_pad(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x)
            return to_device(x)

    out_arr = create_zarr(
        tuple(shape),
        dtype=timelapse.dtype,
        store_or_path=store_or_path,
        overwrite=overwrite,
    )

    if reference_channel is None:
        channel = ...
    else:
        channel = reference_channel

    prev_frame = maybe_pad(timelapse[0])
    out_arr[0] = to_cpu(prev_frame)

    ndi = import_module("scipy", "ndimage", arr=prev_frame)
    skreg = import_module("skimage", "registration", arr=prev_frame)

    for t in tqdm(range(timelapse.shape[0] - 1), "Registration"):
        next_frame = maybe_pad(timelapse[t + 1])
        shift, _, _ = skreg.phase_cross_correlation(
            prev_frame[channel],
            next_frame[channel],
            overlap_ratio=overlap_ratio,
            normalization=normalization,
            **kwargs,
        )

        LOG.info("Shift at {t}: {shift}", t=t, shift=shift)
        print(f"Shift at {t}: {shift}")

        if reference_channel is not None:
            shift = (0, *shift)

        next_frame = ndi.shift(next_frame, shift, order=1)
        out_arr[t + 1] = to_cpu(next_frame)

        prev_frame = next_frame

    return out_arr
