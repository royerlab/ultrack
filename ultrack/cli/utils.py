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


def batch_index_option() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--batch-index",
            "-b",
            required=False,
            default=None,
            show_default=True,
            type=int,
            help="batch index to process a subset of time points. ATTENTION: this it not the time index.",
        )(f)

    return decorator
