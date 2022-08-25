import click

from ultrack import link
from ultrack.cli.utils import config_option, overwrite_option
from ultrack.config import MainConfig


@click.command("link")
@config_option()
@overwrite_option()
def link_cli(config: MainConfig, overwrite: bool) -> None:
    """Links segmentation candidates adjacent in time."""
    link(config.linking_config, config.data_config, overwrite=overwrite)
