from pathlib import Path
from typing import Any, Dict

import pytest
import toml
from pydantic.v1 import ValidationError

from ultrack.config import load_config


def _assert_input_in_target(input: Dict, target: Dict) -> None:
    """Asserts input values are in target recursivelly"""
    for k in input:
        if isinstance(input[k], dict):
            _assert_input_in_target(input[k], target[k])
        else:
            assert target[k] == input[k]


def _format_config(config: Dict) -> None:
    """Formats dictionary config according to Config transforms."""
    config["data_config"] = config.pop("data")
    config["segmentation_config"] = config.pop("segmentation")
    config["linking_config"] = config.pop("linking")
    config["tracking_config"] = config.pop("tracking")


def test_config_content(config_path: Path, config_content: Dict[str, Any]) -> None:
    """Tests if content is loaded correctly"""
    config = load_config(config_path)
    _format_config(config_content)
    _assert_input_in_target(config_content, config.dict())


def test_invalid_config_content(tmp_path: Path, config_content: Dict[str, Any]) -> None:
    """Tests invalid content"""
    config_content["segmentation"]["ws_hierarchy"] = "other"
    path = tmp_path / "config.toml"

    with open(path, mode="w") as f:
        toml.dump(config_content, f)

    with pytest.raises(ValidationError):
        load_config(path)
