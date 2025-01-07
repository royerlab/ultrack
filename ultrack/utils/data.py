import logging
import shutil
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
import skimage.morphology as morph
from skimage.data import binary_blobs
from skimage.segmentation import find_boundaries, relabel_sequential, watershed

LOG = logging.getLogger(__name__)


def make_segmentation_mock_data(
    size: int = 64,
    n_dim: int = 3,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generates mock segmentation dataset producing binary blobs, their contours and labels."""
    if rng is None:
        rng = np.random.default_rng(42)

    blobs = binary_blobs(length=size, n_dim=n_dim, volume_fraction=0.5, rng=rng)

    edt = ndi.distance_transform_edt(blobs)
    markers, _ = ndi.label(morph.h_maxima(edt, 2))
    labels = watershed(-edt, markers, mask=blobs)
    contours = find_boundaries(labels)
    labels, _, _ = relabel_sequential(labels)

    return blobs, contours, labels


def make_config_content(kwargs: Dict[str, Any] = {}) -> Dict[str, Any]:
    """Generates a mock configuration content dictionary."""
    content = {
        "data": {},
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
            "n_threads": 0,
            "link_function": "identity",
            "solver_name": "GUROBI",
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


def make_cell_division_mock_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Short timelapse of a sphere "dividing" into two.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Foreground, contours, labels maps.
    """
    cells = np.zeros((5, 64, 64, 64), dtype=bool)
    ball = morph.ball(radius=5, dtype=bool)

    cells[(0,) + tuple(slice(27, 27 + s) for s in ball.shape)] |= ball

    cells[(1,) + tuple(slice(24, 24 + s) for s in ball.shape)] |= ball  # \
    cells[(1,) + tuple(slice(30, 30 + s) for s in ball.shape)] |= ball  # --- division

    cells[(2,) + tuple(slice(22, 22 + s) for s in ball.shape)] |= ball
    cells[(2,) + tuple(slice(32, 32 + s) for s in ball.shape)] |= ball

    cells[(3,) + tuple(slice(20, 20 + s) for s in ball.shape)] |= ball
    cells[(3,) + tuple(slice(35, 35 + s) for s in ball.shape)] |= ball  # \
    cells[(3,) + tuple(slice(29, 29 + s) for s in ball.shape)] |= ball  # --- division

    cells[(4,) + tuple(slice(18, 18 + s) for s in ball.shape)] |= ball
    cells[(4,) + tuple(slice(37, 37 + s) for s in ball.shape)] |= ball
    cells[(4,) + tuple(slice(27, 27 + s) for s in ball.shape)] |= ball

    contours = np.zeros_like(cells, dtype=np.float32)
    labels = np.zeros_like(cells, dtype=np.int32)

    for t in range(cells.shape[0]):
        edt = ndi.distance_transform_edt(cells[t])
        markers, _ = ndi.label(morph.h_maxima(edt, 2))
        label = watershed(-edt, markers, mask=cells[t])
        contours[t] = find_boundaries(label)
        labels[t] = label

    return cells, contours, labels


def validate_and_overwrite_path(
    path: Path, overwrite: bool, msg_type: Literal["cli", "api"]
) -> None:
    """Validates and errors existance of path (or dir) and overwrites it if requested."""

    if msg_type == "cli":
        msg = f"{path} already exists. Set `--overwrite` option to overwrite it."

    elif msg_type == "api":
        msg = f"{path} already exists. Set `overwrite=True` to overwrite it."

    else:
        raise ValueError(f"Invalid `msg_type` {msg_type}, must be `cli` or `api`.")

    if path.exists():
        if overwrite:
            if path.is_dir():
                shutil.rmtree(path)
            else:
                path.unlink()
        else:
            raise ValueError(msg)
