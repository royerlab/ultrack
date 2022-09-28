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
        with pytest.warns(UserWarning):
            # batch index with overwrite should trigger warning
            _run_command(["track", "-cfg", instance_config_path, "-ow", "-b", "0"])

    def test_summary(self, instance_config_path: str, tmp_path: Path) -> None:
        _run_command(["data_summary", "-cfg", instance_config_path, "-o", tmp_path])

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

    def test_zarr_napari_export(
        self, instance_config_path: str, tmp_path: Path
    ) -> None:
        _run_command(
            [
                "export",
                "zarr-napari",
                "-cfg",
                instance_config_path,
                "-o",
                str(tmp_path / "results"),
            ]
        )

    @pytest.mark.parametrize("mode", ["all", "links", "solutions"])
    def test_clear_database(self, instance_config_path: str, mode: str) -> None:
        _run_command(
            [
                "clear_database",
                mode,
                "-cfg",
                instance_config_path,
            ]
        )


def test_create_config(tmp_path: Path) -> None:
    _run_command(["create_config", str(tmp_path / "config.toml")])


def test_estimate_params(zarr_dataset_paths: List[str], tmp_path: Path) -> None:
    _run_command(["estimate_params", zarr_dataset_paths[2], "-o", str(tmp_path)])


def test_labesl_to_edges(zarr_dataset_paths: List[str], tmp_path: Path) -> None:
    _run_command(
        ["labels_to_edges", zarr_dataset_paths[2], "-o", str(tmp_path / "output")]
    )
