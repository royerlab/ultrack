from pathlib import Path

import click

from ultrack.cli.utils import config_option
from ultrack.config import MainConfig


@click.command()
@click.option("--input-path", "-i", type=click.Path(path_type=Path), required=True)
@config_option()
def initialize(input_path: Path, config: MainConfig) -> None:
    """Initializes tracking model from input data."""
