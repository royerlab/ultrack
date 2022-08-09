import click

from ultrack.cli.utils import config_option
from ultrack.config import MainConfig


@click.command("compute")
@config_option()
def compute_cli(config: MainConfig) -> None:
    """Computes tracking and segmentation."""

    # TODO: check if this function/wrapper is necessary
