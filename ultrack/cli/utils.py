from typing import Callable

import click

from ultrack.config import MainConfig, load_config


def _config_callback(ctx: click.Context, opt: click.Option, value: str) -> MainConfig:
    return load_config(value)


def config_option() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--config",
            "-cfg",
            required=True,
            help="ultrack configuration file (.toml)",
            callback=_config_callback,
        )(f)

    return decorator
