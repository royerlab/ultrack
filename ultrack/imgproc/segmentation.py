import logging
from typing import Optional

import edt
import numpy as np
from numpy.typing import ArrayLike
from skimage.morphology import reconstruction

from ultrack.imgproc.utils import _channel_iterator, _parse_voxel_size
from ultrack.utils.constants import ULTRACK_DEBUG
from ultrack.utils.cuda import import_module, is_cupy_array, to_cpu

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
    ndi = import_module("scipy", "ndimage", arr=mask)

    seed = np.minimum(seed, mask, out=seed)  # just making sure

    for _ in range(iterations):
        seed = ndi.grey_dilation(seed, size=3, output=seed, mode="constant")
        seed = np.minimum(seed, mask, out=seed)

    return seed


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
        if is_cupy_array(_image):
            background = reconstruction_by_dilation(seed, _image, 100)
        else:
            if seed.ndim > 2:
                LOG.warning(
                    "Using CPU background reconstruction, this could take a while, consider using GPU for 3D images."
                )
            np.minimum(seed, _image, out=seed)  # required condition for reconstruction
            background = reconstruction(seed, _image, method="dilation")
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


def inverted_edt(
    mask: ArrayLike,
    voxel_size: Optional[ArrayLike] = None,
    axis: Optional[int] = None,
) -> ArrayLike:
    """
    Computes Euclidean distance transform (EDT), inverts and normalizes it.

    Parameters
    ----------
    mask : ArrayLike
        Cells' foreground mask.
    voxel_size : Optional[ArrayLike], optional
        Voxel size, by default None
    axis : Optional[int], optional
        Axis to compute the EDT, by default None

    Returns
    -------
    ArrayLike
        Inverted and normalized EDT.
    """
    mask = np.asarray(mask)
    if axis is None:
        dist = edt.edt(mask, anisotropy=voxel_size)
    else:
        dist = np.stack(
            [
                edt.edt(np.take(mask, i, axis=axis), anisotropy=voxel_size)
                for i in range(mask.shape[axis])
            ],
            axis=axis,
        )
    dist = dist / dist.max()
    dist = 1.0 - dist
    dist[mask == 0] = 1
    return dist


class Cellpose:
    def __init__(self, **kwargs) -> None:
        """See cellpose.models.Cellpose documentation for details."""
        from cellpose.models import CellposeModel as _Cellpose

        if "pretrained_model" not in kwargs and "model_type" not in kwargs:
            kwargs["model_type"] = "cyto"

        self.model = _Cellpose(**kwargs)

    def __call__(self, image: ArrayLike, **kwargs) -> np.ndarray:
        """
        Predicts image labels.
        See cellpose.models.Cellpose.eval documentation for details.
        """
        labels, _, _ = self.model.eval(image, **kwargs)
        return labels
