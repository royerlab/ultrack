from pathlib import Path
from typing import Any, Dict

import pytest
import toml


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
    path = tmp_path / "config"
    with open(path, mode="w") as f:
        toml.dump(config_content, f)
    return path
