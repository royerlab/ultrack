import tempfile
from pathlib import Path
from typing import List

import pytest
import toml
import zarr

from ultrack.cli.main import main
from ultrack.config import load_config
from ultrack.utils.data import make_config_content


def _run_command(command_and_args: List[str]) -> None:
    try:
        main(command_and_args)
    except SystemExit as exit:
        assert exit.code == 0


@pytest.mark.usefixtures("zarr_dataset_paths")
@pytest.mark.parametrize(
    "instance_config_path",
    [
        {},  # defaults
        {
            "segmentation.n_workers": 2,
            "linking.n_workers": 2,
            "data.n_workers": 2,
        },
    ],
    indirect=True,
)
class TestCommandLine:
    @pytest.fixture(scope="class")
    def instance_config_path(self, request) -> str:
        """Created this fixture so configuration are shared between methods."""
        with tempfile.TemporaryDirectory() as tmpdir:
            kwargs = {"data.working_dir": tmpdir}
            if hasattr(request, "param"):
                kwargs.update(request.param)

            content = make_config_content(kwargs)
            config_path = f"{tmpdir}/config.toml"
            with open(config_path, mode="w") as f:
                toml.dump(content, f)
            yield config_path

    def test_segment(
        self, instance_config_path: str, zarr_dataset_paths: List[str]
    ) -> None:
        _run_command(
            [
                "segment",
                "-cfg",
                instance_config_path,
                "-dl",
                "detection",
                "-el",
                "edges",
            ]
            + zarr_dataset_paths
        )

    def test_add_shift(self, instance_config_path: str) -> None:
        config = load_config(instance_config_path)
        tmp_store = zarr.TempStore(suffix=".zarr")
        zarr.zeros((2,) + tuple(config.data_config.metadata["shape"]), store=tmp_store)
        _run_command(["add_shift", "-cfg", str(instance_config_path), tmp_store.path])

    def test_link_iou(self, instance_config_path: str) -> None:
        _run_command(["link", "-cfg", str(instance_config_path)])

    def test_link_dct(
        self, instance_config_path: str, zarr_dataset_paths: List[str]
    ) -> None:
        # using detection and edges layer to simulate image channel
        _run_command(
            ["link", "-cfg", str(instance_config_path), "-ow"] + zarr_dataset_paths[:2]
        )

    def test_solve(self, instance_config_path: str) -> None:
        with pytest.warns(UserWarning):
            # batch index with overwrite should trigger warning
            _run_command(["solve", "-cfg", instance_config_path, "-ow", "-b", "0"])

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
                "0.5,1,1",
                "-ma",
                "5",
                "-di",
                "1",
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
                "--include-parents",
            ]
        )

    @pytest.mark.parametrize("mode", ["solutions", "links", "all"])
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


def test_labels_to_edges(zarr_dataset_paths: List[str], tmp_path: Path) -> None:
    _run_command(
        ["labels_to_edges", zarr_dataset_paths[2], "-o", str(tmp_path / "output")]
    )
