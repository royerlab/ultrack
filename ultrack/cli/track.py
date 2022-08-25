from typing import Optional

import click

from ultrack import track
from ultrack.cli.utils import batch_index_option, config_option, overwrite_option
from ultrack.config import MainConfig


@click.command("track")
@config_option()
@batch_index_option()
@overwrite_option()
def track_cli(config: MainConfig, batch_index: Optional[int], overwrite: bool) -> None:
    """Compute tracks by selecting optimal linking between candidate segments."""
    track(config.tracking_config, config.data_config, batch_index, overwrite)
