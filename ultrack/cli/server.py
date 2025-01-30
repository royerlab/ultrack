from typing import Optional

import click

from ultrack import MainConfig
from ultrack.api import start_server
from ultrack.cli.utils import config_option


@click.command("server")
@click.option("--host", help="Host address", default="0.0.0.0", show_default=True)
@click.option(
    "--port",
    help="Port number to listen on",
    type=int,
    default=61234,
    show_default=True,
)
@click.option(
    "--api-results-path",
    help="Path to the API results folder",
    type=str,
    default=None,
    show_default=True,
)
@config_option()
def server_cli(
    host: str, port: int, api_results_path: Optional[str], config: MainConfig
) -> None:
    """Start the websockets ultrack API."""
    start_server(api_results_path, ultrack_data_config=config, host=host, port=port)
