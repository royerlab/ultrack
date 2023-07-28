import logging
from typing import List, Optional

import numpy as np
from numpy.typing import ArrayLike

from ultrack.utils.constants import ULTRACK_DEBUG
from ultrack.utils.cuda import import_module, to_cpu

LOG = logging.getLogger(__name__)

try:
    import cupy as xp

except ImportError as e:
    LOG.info(e)
    LOG.info("cupy not found, using CPU processing")
    import numpy as xp


def reconstruction_by_dilation(
    seed: ArrayLike, mask: ArrayLike, iterations: int
) -> ArrayLike:
    """
    Morphological reconstruction by dilation.
    This function does not compute the full reconstruction.
    The reconstruction is garanteed to be computed fully if the
    number of iterations is equal to the number of pixels.
    See: scikit-image docs for details.

    Parameters
    ----------
    seed : ArrayLike
        Dilation seeds.
    mask : ArrayLike
        Guidance mask.
    iterations : int
        Number of dilations.

    Returns
    -------
        Image reconstructed by dilation.
    """
    ndi = import_module("scipy", "ndimage")

    seed = np.minimum(seed, mask, out=seed)  # just making sure

    for _ in range(iterations):
        seed = ndi.grey_dilation(seed, size=3, output=seed, mode="constant")
        seed = np.minimum(seed, mask, out=seed)

    return seed


def _parse_voxel_size(
    voxel_size: Optional[ArrayLike],
    ndim: int,
    channel_axis: Optional[int],
) -> np.ndarray:
    """Parses voxel size and returns it in the correct format."""
    ndim = ndim - (channel_axis is not None)

    if voxel_size is None:
        voxel_size = (1.0,) * ndim

    if len(voxel_size) != ndim:
        raise ValueError(
            "Voxel size must have the same number of elements as the image dimensions."
            f"Expected {ndim} got {len(voxel_size)}."
        )

    return np.asarray(voxel_size)


def _channel_iterator(
    image: ArrayLike,
    channel_axis: Optional[int],
) -> List[Optional[int]]:
    """Iterates over channels, it returns a [None] if channel_axis is None."""

    if channel_axis is None:
        channel_iterator = [None]
    else:
        channel_iterator = range(image.shape[channel_axis])

    return channel_iterator


def detect_foreground(
    image: ArrayLike,
    voxel_size: Optional[ArrayLike] = None,
    sigma: float = 15.0,
    remove_hist_mode: bool = False,
    min_foreground: float = 0.0,
    channel_axis: Optional[int] = None,
) -> np.ndarray:
    """
    Detect foreground using morphological reconstruction by dilation and thresholding.

    Parameters
    ----------
    image : ArrayLike
        Input image.
    voxel_size : ArrayLike
        Array of voxel size (z, y, x).
    sigma : float, optional
        Sigma used to estimate background, it will be divided by voxel size, by default 15.0
    remove_hist_mode : bool, optional
        Removes histogram mode before computing otsu threshold, useful when background regions are being detected.
    min_foreground : float, optional
        Minimum value of foreground pixels after background subtraction and smoothing, by default 0.0
    channel_axis : Optional[int], optional
        When provided it will be used to compute the foreground mask for each channel separately and merge them.

    Returns
    -------
    ArrayLike
        Binary foreground mask.
    """
    ndi = import_module("scipy", "ndimage")
    filters = import_module("skimage", "filters")
    exposure = import_module("skimage", "exposure")

    voxel_size = _parse_voxel_size(voxel_size, image.ndim, channel_axis)

    shape = list(image.shape)
    if channel_axis is not None:
        shape.pop(channel_axis)

    sigmas = sigma / voxel_size
    LOG.info(f"Detecting foreground with voxel size {voxel_size} and sigma {sigma}")
    LOG.info(f"Sigmas after scaling {sigmas}")

    output = np.zeros(shape, dtype=bool)

    for ch in _channel_iterator(image, channel_axis):
        if ch is None:
            _image = image
        else:
            _image = np.take(image, indices=ch, axis=channel_axis)

        _image = xp.asarray(_image)

        seed = ndi.gaussian_filter(_image, sigma=sigmas)
        background = reconstruction_by_dilation(seed, _image, 100)
        del seed

        foreground = _image - background
        del background

        # threshold in smaller image to save memory and sqrt to deskew data distribution towards left
        small_foreground = np.sqrt(
            ndi.zoom(foreground, (0.25,) * foreground.ndim, order=1)
        )

        # begin thresholding
        robust_max = np.quantile(small_foreground, 1 - 1e-6)
        small_foreground = np.minimum(
            robust_max, small_foreground, out=small_foreground
        )

        # number of bins according to maximum value
        nbins = int(robust_max / 10)  # binning with window of 10
        nbins = min(nbins, 256)
        nbins = max(nbins, 10)

        LOG.info(f"Estimated almost max. {np.square(robust_max)}")
        LOG.info(f"Histogram with {nbins}")

        hist, bin_centers = exposure.histogram(small_foreground, nbins)

        # histogram disconsidering pixels we are sure are background
        if remove_hist_mode:
            remaining_background_idx = hist.argmax() + 1
            hist = hist[remaining_background_idx:]
            bin_centers = bin_centers[remaining_background_idx:]

        del small_foreground
        threshold = np.square(filters.threshold_otsu(hist=(hist, bin_centers)))
        LOG.info(f"Threshold {threshold}")

        threshold = max(threshold, min_foreground)
        LOG.info(f"Threshold after minimum filtering {threshold}")

        mask = foreground > threshold
        del foreground

        struct = ndi.generate_binary_structure(mask.ndim, 2)
        mask = ndi.binary_opening(mask, structure=struct, output=mask)
        # opening but with border value=True on erosion
        mask = ndi.binary_dilation(mask, structure=struct, output=mask)
        mask = ndi.binary_erosion(
            mask, structure=struct, output=mask, border_value=True
        )
        output |= to_cpu(mask)

    if ULTRACK_DEBUG:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(to_cpu(_image))
        viewer.add_labels(output)
        napari.run()

    return output


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
        ndi.gaussian_filter(_image, sigma=sigmas, output=_image)

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

        im_max = im_max.astype(np.float32)
        _image = (1 / im_max) * norm_factor * _image
        LOG.info(f"Maximum {im_max}")

        np.subtract(norm_factor, _image, out=_image)  # inverting
        np.clip(_image, 0, norm_factor, out=_image)

        output += to_cpu(_image)

    return output
