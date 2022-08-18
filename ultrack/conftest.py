import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import toml
import zarr

from ultrack.config.config import MainConfig, load_config
from ultrack.core.segmentation.processing import segment
from ultrack.utils.data import make_segmentation_mock_data

LOG = logging.getLogger(__name__)


@pytest.fixture
def config_content(tmp_path: Path, request) -> Dict[str, Any]:
    content = {
        "data": {"working_dir": str(tmp_path)},
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
            "edge_transform": None,
        },
    }

    if hasattr(request, "param"):
        for keys, value in request.param.items():
            param = content
            keys = keys.split(".")
            for k in keys[:-1]:
                param = param[k]
            param[keys[-1]] = value

    LOG.info(content)
    return content


@pytest.fixture
def config_path(tmp_path: Path, config_content: Dict[str, Any]) -> Path:
    path = tmp_path / "config.toml"
    with open(path, mode="w") as f:
        toml.dump(config_content, f)
    return path


@pytest.fixture
def config_instance(config_path: Path) -> MainConfig:
    return load_config(config_path)


@pytest.fixture
def zarr_dataset_paths(tmp_path: Path) -> List[str]:

    paths = []
    for filename in ("detection.zarr", "edge.zarr"):
        path = tmp_path / filename
        store = zarr.DirectoryStore(path)
        _ = zarr.zeros(
            shape=(25, 128, 128, 128), chunks=(1, 128, 128, 128), store=store
        )
        paths.append(str(path))

    return paths


@pytest.fixture
def segmentation_mock_data(request) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return make_segmentation_mock_data(**request.param)


@pytest.fixture
def timelapse_mock_data(request) -> Tuple[zarr.Array, zarr.Array]:
    if not hasattr(request, "param"):
        request.param = {}

    length = request.param.pop("length", 4)
    blobs, contours, _ = make_segmentation_mock_data(**request.param)
    shape = (length,) + blobs.shape

    detection = zarr.empty(
        shape, store=zarr.MemoryStore(), chunks=(1, *blobs.shape), dtype=blobs.dtype
    )
    edge = zarr.empty(
        shape, store=zarr.MemoryStore(), chunks=(1, *blobs.shape), dtype=contours.dtype
    )

    for t in range(length):
        detection[t] = blobs
        edge[t] = contours

    return detection, edge


@pytest.fixture
def segmentation_database_mock_data(
    config_instance: MainConfig,
    timelapse_mock_data: Tuple[zarr.Array, zarr.Array],
) -> MainConfig:
    detection, edge = timelapse_mock_data
    segment(
        detection,
        edge,
        config_instance.segmentation_config,
        config_instance.data_config,
    )
    return config_instance
