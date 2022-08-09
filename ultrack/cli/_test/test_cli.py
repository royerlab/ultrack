from pathlib import Path
from typing import Sequence

from ultrack.cli.main import main


def _run_command(commands: Sequence[str]) -> None:
    try:
        main(commands)
    except SystemExit as exit:
        assert exit.code == 0


def test_initialize(config_path: Path) -> None:
    _run_command(["initialize", "-cfg", str(config_path), "-i", "fake.zarr"])


def test_compute(config_path: Path) -> None:
    _run_command(["compute", "-cfg", str(config_path)])


def test_export(config_path: Path) -> None:
    _run_command(["export", "-cfg", str(config_path), "-f", "ctc"])
