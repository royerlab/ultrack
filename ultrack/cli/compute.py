import click

from ultrack.cli.utils import config_option
from ultrack.config import MainConfig


@click.command()
@config_option()
def compute(config: MainConfig) -> None:
    """Computes tracking and segmentation."""
