import logging
from typing import Any, Dict, Tuple

import numpy as np
import scipy.ndimage as ndi
from skimage.data import binary_blobs
from skimage.morphology import h_maxima
from skimage.segmentation import find_boundaries, relabel_sequential, watershed

LOG = logging.getLogger(__name__)


def make_segmentation_mock_data(
    size: int = 64,
    n_dim: int = 3,
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


def make_config_content(kwargs: Dict[str, Any] = {}) -> Dict[str, Any]:
    """Generates a mock configuration content dictionary."""
    content = {
        "data": {},
        "reader": {},
        "segmentation": {
            "threshold": 0.5,
            "max_area": 7500,
            "min_area": 500,
            "min_frontier": 0.1,
            "anisotropy_penalization": 0.0,
            "ws_hierarchy": "area",
            "n_workers": 1,
        },
        "linking": {
            "max_neighbors": 10,
            "max_distance": 15.0,
            "n_workers": 1,
        },
        "tracking": {
            "appear_weight": -0.2,
            "disappear_weight": -1.0,
            "division_weight": -0.1,
            "dismiss_weight_guess": None,
            "include_weight_guess": None,
            "solution_gap": 0.001,
            "time_limit": 36000,
            "method": -1,
            "n_threads": -1,
            "link_function": "identity",
        },
    }

    for keys, value in kwargs.items():
        param = content
        keys = keys.split(".")
        for k in keys[:-1]:
            param = param[k]
        param[keys[-1]] = value

    LOG.info(content)

    return content
