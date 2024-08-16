import logging
from typing import Optional

import numpy as np
from numpy.typing import ArrayLike

from ultrack.imgproc.utils import _channel_iterator, _parse_voxel_size
from ultrack.utils.cuda import import_module, to_cpu

LOG = logging.getLogger(__name__)

try:
    import cupy as xp

except ImportError as e:
    LOG.info(e)
    LOG.info("cupy not found, using CPU processing")
    import numpy as xp


def normalize(
    image: ArrayLike,
    gamma: float,
    lower_q: float = 0.001,
    upper_q: float = 0.9999,
) -> ArrayLike:
    """
    Normalize image to between [0, 1] and applies a gamma transform (x ^ gamma).

    Parameters
    ----------
    image : ArrayLike
        Images as an T,Y,X,C array.
    gamma : float
        Expoent of gamma transform.
    lower_q : float, optional
        Lower quantile for normalization.
    upper_q : float, optional
        Upper quantile for normalization.

    Returns
    -------
    ArrayLike
        Normalized array.
    """
    frame = image
    frame = frame - np.quantile(frame, lower_q)
    frame = frame / np.quantile(frame, upper_q)
    frame = np.clip(frame, 0, 1)

    if gamma != 1.0:
        frame = np.power(frame, gamma)

    return frame


def robust_invert(
    image: ArrayLike,
    voxel_size: Optional[ArrayLike] = None,
    sigma: float = 1.0,
    lower_quantile: Optional[float] = None,
    upper_quantile: Optional[float] = 0.9999,
    channel_axis: Optional[int] = None,
) -> np.ndarray:
    """
    Inverts an image robustly by first smoothing it with a gaussian filter
    and then normalizing it to [0, 1].

    Parameters
    ----------
    image : ArrayLike
        Input image.
    voxel_size : ArrayLike
        Array of voxel size (z, y, x).
    sigma : float, optional
        Sigma used to smooth the image, by default 1.0.
    lower_quantile : Optional[float], optional
        Lower quantile used to clip the intensities, minimum used when None.
    upper_quantile : Optional[float], optional
        Upper quantile used to clip the intensities, maximum used when None.
    channel_axis : Optional[int], optional
        When provided it will invert each channel separately and merge them.

    Returns
    -------
    ArrayLike
        Inverted and normalized image.
    """
    LOG.info(f"Channel axis {channel_axis}")

    ndi = import_module("scipy", "ndimage")

    shape = list(image.shape)
    if channel_axis is not None:
        shape.pop(channel_axis)
        LOG.info(f"Shape={shape} after removing channel axis={channel_axis}")

    voxel_size = _parse_voxel_size(voxel_size, image.ndim, channel_axis)
    sigmas = sigma / voxel_size
    LOG.info(f"Inverting with voxel size {voxel_size} and sigma {sigma}")
    LOG.info(f"Sigmas after scaling {sigmas}")

    iterator = _channel_iterator(image, channel_axis)
    output = np.zeros(shape, dtype=np.float32)
    norm_factor = 1 / len(iterator)

    for ch in iterator:
        if ch is None:
            _image = image
        else:
            _image = np.take(image, indices=ch, axis=channel_axis)

        _image = xp.asarray(_image)
        _image = ndi.gaussian_filter(_image, sigma=sigmas)

        flat_small_img = ndi.zoom(_image, (0.25,) * _image.ndim, order=1).ravel()

        if lower_quantile is not None:
            im_min = np.quantile(flat_small_img, lower_quantile)
        else:
            im_min = flat_small_img.min()

        im_min = im_min.astype(_image.dtype)
        # safe sub, it could be unsigned
        np.clip(_image, im_min, None, out=_image)
        _image -= im_min
        LOG.info(f"Minimum {im_min}")

        if upper_quantile is not None:
            im_max = np.quantile(flat_small_img, upper_quantile)
        else:
            im_max = flat_small_img.max()
        del flat_small_img

        im_max = np.maximum(im_max.astype(np.float32), 1e-8)
        _image = (1 / im_max) * norm_factor * _image
        LOG.info(f"Maximum {im_max}")

        np.subtract(norm_factor, _image, out=_image)  # inverting
        np.clip(_image, 0, norm_factor, out=_image)

        output += to_cpu(_image)

    return output
