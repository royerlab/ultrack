from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pytest
import toml
import zarr

from ultrack.config.config import MainConfig, load_config
from ultrack.config.dataconfig import DatabaseChoices
from ultrack.core.linking.processing import link
from ultrack.core.segmentation.processing import segment
from ultrack.core.solve.processing import solve
from ultrack.utils.data import (
    make_cell_division_mock_data,
    make_config_content,
    make_segmentation_mock_data,
)

# from testing.postgresql import Postgresql


@pytest.fixture
def config_content(tmp_path: Path, request) -> Dict[str, Any]:
    kwargs = {"data.working_dir": str(tmp_path)}
    if hasattr(request, "param"):
        kwargs.update(request.param)

    # FIXME: needs to fork testing.postgresql
    # if postgresql create dummy server and close when done
    is_postgresql = kwargs.get("data.database") == DatabaseChoices.postgresql.value

    if is_postgresql:
        # FIXME: not working, falling back to sqlite
        kwargs["data.database"] = DatabaseChoices.sqlite.value

        # if platform.system() == "Windows":
        #     pytest.skip("Skipping postgresql testing on Windows")

        # postgresql = Postgresql()
        # kwargs["data.address"] = postgresql.url().split("//")[1]

    yield make_config_content(kwargs)

    # if is_postgresql:
    #     postgresql.stop()


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
        timelapse_mock_data, ("foreground.zarr", "contours.zarr", "labels.zarr")
    ):
        path = tmp_path / filename
        dst_store = zarr.NestedDirectoryStore(path)
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
    length = kwargs.pop("length", 3)

    blobs, contours, labels = make_segmentation_mock_data(**kwargs)
    shape = (length,) + blobs.shape

    foreground = zarr.empty(
        shape, store=zarr.MemoryStore(), chunks=(1, *blobs.shape), dtype=blobs.dtype
    )
    edge = zarr.empty(
        shape, store=zarr.MemoryStore(), chunks=(1, *blobs.shape), dtype=contours.dtype
    )
    segmentation = zarr.empty(
        shape, store=zarr.MemoryStore(), chunks=(1, *blobs.shape), dtype=labels.dtype
    )

    for t in range(length):
        foreground[t] = blobs
        edge[t] = contours
        segmentation[t] = labels

    return foreground, edge, segmentation


@pytest.fixture
def segmentation_database_mock_data(
    config_instance: MainConfig,
    timelapse_mock_data: Tuple[zarr.Array, zarr.Array, zarr.Array],
) -> MainConfig:
    foreground, edge, _ = timelapse_mock_data
    segment(
        foreground,
        edge,
        config_instance,
    )
    return config_instance


@pytest.fixture(scope="function")
def linked_database_mock_data(
    segmentation_database_mock_data: MainConfig,
) -> MainConfig:
    config = segmentation_database_mock_data
    link(config)
    return config


@pytest.fixture(scope="function")
def tracked_database_mock_data(
    linked_database_mock_data: MainConfig,
) -> MainConfig:
    config = linked_database_mock_data
    solve(config)
    return config


@pytest.fixture
def cell_division_mock_data() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    return make_cell_division_mock_data()


@pytest.fixture(scope="function")
def tracked_cell_division_mock_data(
    cell_division_mock_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    config_instance: MainConfig,
) -> MainConfig:
    foreground, contours, _ = cell_division_mock_data

    segment(
        foreground,
        contours,
        config_instance,
    )
    link(config_instance)
    solve(config_instance)

    return config_instance
