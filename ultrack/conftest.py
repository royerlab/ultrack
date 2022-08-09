from pathlib import Path
from typing import Any, Dict, List

import pytest
import toml
import zarr


@pytest.fixture
def config_content() -> Dict[str, Any]:
    content = {
        "working_dir": ".",
        "reader": {},
        "init": {
            "threshold": 0.5,
            "max_area": 7500,
            "min_area": 500,
            "min_frontier": 0.1,
            "anisotropy_penalization": 0.0,
            "ws_hierarchy": "area",
            "n_workers": 1,
            "max_neighbors": 10,
            "max_distance": 15.0,
        },
        "compute": {
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
    return content


@pytest.fixture
def config_path(tmp_path: Path, config_content: Dict[str, Any]) -> Path:
    path = tmp_path / "config.toml"
    with open(path, mode="w") as f:
        toml.dump(config_content, f)
    return path


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
