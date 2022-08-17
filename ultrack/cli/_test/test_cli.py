from pathlib import Path
from typing import List

from ultrack.cli.main import main


def _run_command(commands: List[str]) -> None:
    try:
        main(commands)
    except SystemExit as exit:
        assert exit.code == 0


def test_segment(config_path: Path, zarr_dataset_paths: List[str]) -> None:
    _run_command(["segment", "-cfg", str(config_path)] + zarr_dataset_paths)


def test_compute(config_path: Path) -> None:
    _run_command(["compute", "-cfg", str(config_path)])


def test_export(config_path: Path) -> None:
    _run_command(["export", "-cfg", str(config_path), "-f", "ctc"])
