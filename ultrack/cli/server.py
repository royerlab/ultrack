import os

import click
import uvicorn

from ultrack.api.app import app


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
def server_cli(host: str, port: int, api_results_path: str) -> None:
    """Start the websockets ultrack API."""
    if api_results_path is not None:
        os.environ["API_RESULTS_PATH"] = str(api_results_path)

    uvicorn.run(app, host=host, port=port)
