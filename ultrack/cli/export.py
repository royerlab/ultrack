import click

from ultrack.cli.utils import config_option
from ultrack.config import MainConfig


@click.command()
@config_option()
@click.option(
    "format",
    "-f",
    type=click.Choice(["mamut", "ctc"]),
    help="Output format.",
    required=True,
)
def export(config: MainConfig, format: str) -> None:
    """Exports tracking and segmentation results to selected format."""
