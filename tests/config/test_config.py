from pathlib import Path
from typing import Any, Dict

import pytest
import toml
from pydantic import ValidationError

from ultrack.config import load_config
from ultrack.config.config import NAME_TO_WS_HIER


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
    return _config_path(tmp_path, config_content)


def _config_path(tmp_path: Path, config_content: Dict[str, Any]) -> Path:
    file_path = tmp_path / "config.toml"

    with open(file_path, "w") as f:
        toml.dump(config_content, f)

    return file_path


def _assert_input_in_target(input: Dict, target: Dict) -> None:
    """Asserts input values are in target recursivelly"""
    for k in input:
        if isinstance(input[k], dict):
            _assert_input_in_target(input[k], target[k])
        else:
            assert target[k] == input[k]


def _format_config(config: Dict) -> None:
    """Formats dictionary config according to Config transforms."""
    config["init"]["ws_hierarchy"] = NAME_TO_WS_HIER[config["init"]["ws_hierarchy"]]

    config["working_dir"] = Path(config["working_dir"])

    config["reader_config"] = config.pop("reader")
    config["init_config"] = config.pop("init")
    config["compute_config"] = config.pop("compute")


def test_config_content(config_path: Path, config_content: Dict[str, Any]) -> None:
    """Tests if content is loaded correctly"""
    config = load_config(config_path)
    _format_config(config_content)
    _assert_input_in_target(config_content, config.dict())


def test_invalid_config_content(tmp_path: Path, config_content: Dict[str, Any]) -> None:
    """Tests invalid content"""
    config_content["init"]["ws_hierarchy"] = "other"
    path = _config_path(tmp_path, config_content)

    with pytest.raises(ValidationError):
        load_config(path)
