from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import toml
import zarr

from ultrack.config.config import MainConfig, load_config
from ultrack.core.linking.processing import link
from ultrack.core.segmentation.processing import segment
from ultrack.core.tracking.processing import track
from ultrack.utils.data import make_config_content, make_segmentation_mock_data


@pytest.fixture
def config_content(tmp_path: Path, request) -> Dict[str, Any]:
    kwargs = {"data.working_dir": str(tmp_path)}
    if hasattr(request, "param"):
        kwargs.update(request.param)
    return make_config_content(kwargs)


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
def zarr_dataset_paths(
    tmp_path: Path, timelapse_mock_data: Tuple[zarr.Array, zarr.Array]
) -> List[str]:

    paths = []
    for src_array, filename in zip(
        timelapse_mock_data, ("detection.zarr", "edge.zarr", "labels.zarr")
    ):
        path = tmp_path / filename
        dst_store = zarr.DirectoryStore(path)
        zarr.copy_store(src_array.store, dst_store)
        paths.append(str(path))

    return paths


@pytest.fixture
def segmentation_mock_data(request) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return make_segmentation_mock_data(**request.param)


@pytest.fixture
def timelapse_mock_data(request) -> Tuple[zarr.Array, zarr.Array, zarr.Array]:
    if not hasattr(request, "param"):
        request.param = {}

    # avoiding popping from instance
    kwargs = request.param.copy()
    length = kwargs.pop("length", 4)

    blobs, contours, labels = make_segmentation_mock_data(**kwargs)
    shape = (length,) + blobs.shape

    detection = zarr.empty(
        shape, store=zarr.MemoryStore(), chunks=(1, *blobs.shape), dtype=blobs.dtype
    )
    edge = zarr.empty(
        shape, store=zarr.MemoryStore(), chunks=(1, *blobs.shape), dtype=contours.dtype
    )
    segmentation = zarr.empty(
        shape, store=zarr.MemoryStore(), chunks=(1, *blobs.shape), dtype=labels.dtype
    )

    for t in range(length):
        detection[t] = blobs
        edge[t] = contours
        segmentation[t] = labels

    return detection, edge, segmentation


@pytest.fixture
def segmentation_database_mock_data(
    config_instance: MainConfig,
    timelapse_mock_data: Tuple[zarr.Array, zarr.Array],
) -> MainConfig:
    detection, edge, _ = timelapse_mock_data
    segment(
        detection,
        edge,
        config_instance.segmentation_config,
        config_instance.data_config,
    )
    return config_instance


@pytest.fixture(scope="function")
def linked_database_mock_data(
    segmentation_database_mock_data: MainConfig,
) -> MainConfig:
    config = segmentation_database_mock_data
    link(config.linking_config, config.data_config)
    return config


@pytest.fixture(scope="function")
def tracked_database_mock_data(
    linked_database_mock_data: MainConfig,
) -> MainConfig:
    config = linked_database_mock_data
    track(config.tracking_config, config.data_config)
    return config
