import tempfile
from pathlib import Path
from typing import List

import pytest
import toml

from ultrack.cli.main import main
from ultrack.utils.data import make_config_content


def _run_command(command_and_args: List[str]) -> None:
    try:
        main(command_and_args)
    except SystemExit as exit:
        assert exit.code == 0


@pytest.mark.usefixtures("zarr_dataset_paths")
class TestCommandLine:
    @pytest.fixture(scope="class")
    def instance_config_path(self) -> str:
        """Created this fixture so configuration are shared between methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            content = make_config_content({"data.working_dir": tmpdir})
            config_path = f"{tmpdir}/config.toml"
            with open(config_path, mode="w") as f:
                toml.dump(content, f)
            yield config_path

    def test_segment(
        self, instance_config_path: str, zarr_dataset_paths: List[str]
    ) -> None:
        _run_command(["segment", "-cfg", instance_config_path] + zarr_dataset_paths)

    def test_link(self, instance_config_path: str) -> None:
        _run_command(["link", "-cfg", str(instance_config_path)])

    def test_tracking(self, instance_config_path: str) -> None:
        _run_command(["track", "-cfg", instance_config_path])

    def test_ctc_export(self, instance_config_path: str, tmp_path: Path) -> None:
        _run_command(
            [
                "export",
                "ctc",
                "-cfg",
                instance_config_path,
                "-s",
                "1,1,1",
                "-o",
                str(tmp_path / "01_RES"),
            ]
        )


def test_create_config(tmp_path: Path) -> None:
    _run_command(["create_config", str(tmp_path / "config.toml")])
