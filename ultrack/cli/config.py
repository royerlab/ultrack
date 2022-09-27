from pathlib import Path

import click
import toml

from ultrack.config.config import MainConfig


@click.command("create_config")
@click.argument("output_path", type=click.Path(path_type=Path), default="config.toml")
def config_cli(output_path: Path) -> None:
    """Creates a configuration file with default values."""
    config = MainConfig()

    if output_path.exists():
        raise ValueError(f"{output_path} already exists.")

    with open(output_path, mode="w") as f:
        toml.dump(config.dict(by_alias=True), f)
