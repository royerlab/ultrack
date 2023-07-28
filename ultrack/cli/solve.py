from typing import Optional

import click

from ultrack import solve
from ultrack.cli.utils import batch_index_option, config_option, overwrite_option
from ultrack.config import MainConfig


@click.command("solve")
@config_option()
@batch_index_option()
@overwrite_option()
def solve_cli(config: MainConfig, batch_index: Optional[int], overwrite: bool) -> None:
    """Compute tracks by selecting optimal linking between candidate segments."""
    solve(config, batch_index, overwrite)
