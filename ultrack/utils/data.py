from typing import Tuple

import numpy as np
import scipy.ndimage as ndi
from skimage.data import binary_blobs
from skimage.morphology import h_maxima
from skimage.segmentation import find_boundaries, relabel_sequential, watershed


def make_segmentation_mock_data(
    size: int, n_dim: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates mock segmentation dataset producing binary blobs, their contours and labels."""
    rng = np.random.default_rng(42)
    blobs = binary_blobs(length=size, n_dim=n_dim, volume_fraction=0.5, seed=rng)

    edt = ndi.distance_transform_edt(blobs)
    markers, _ = ndi.label(h_maxima(edt, 2))
    labels = watershed(-edt, markers, mask=blobs)
    contours = find_boundaries(labels)
    labels, _, _ = relabel_sequential(labels)

    return blobs, contours, labels
