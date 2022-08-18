import tempfile
from pathlib import Path
from typing import List

import pytest
import toml

from ultrack.cli.main import main
from ultrack.utils.data import make_config_content


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

    def _run_command(self, command_and_args: List[str]) -> None:
        try:
            main(command_and_args)
        except SystemExit as exit:
            assert exit.code == 0

    def test_segment(
        self, instance_config_path: str, zarr_dataset_paths: List[str]
    ) -> None:
        self._run_command(
            ["segment", "-cfg", instance_config_path] + zarr_dataset_paths
        )

    def test_link(self, instance_config_path: Path) -> None:
        self._run_command(["link", "-cfg", str(instance_config_path)])

    def test_export(self, instance_config_path: str) -> None:
        self._run_command(["export", "-cfg", instance_config_path, "-f", "ctc"])

    # def test_tracking(config_path: Path) -> None:
    #     _run_command(["track", "-cfg", str(config_path)])
