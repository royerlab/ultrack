from typing import Any, Callable, Optional, Tuple

import click
from toolz import curry

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


def overwrite_option() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--overwrite",
            "-ow",
            is_flag=True,
            default=False,
            type=bool,
            help="allows overwriting existing data.",
        )(f)

    return decorator


@curry
def tuple_callback(
    ctx: click.Context,
    opt: click.Option,
    value: str,
    dtype: Callable = int,
    length: Optional[int] = None,
) -> Optional[Tuple[Any]]:
    """Parses string to tuple given dtype and optional length.
       Returns None if None is supplied.
    Parameters
    ----------
    ctx : click.Context
        CLI context, not used.
    opt : click.Option
        CLI option, not used.
    value : str
        Input value.
    dtype : Callable, optional
        Data type for type casting, by default int
    length : Optional[int], optional
        Optional length for length checking, by default None
    Returns
    -------
    Tuple[Any]
        Tuple of given dtype and length (optional).
    """
    if value is None:
        return None
    tup = tuple(dtype(s) for s in value.split(","))
    if length is not None and length != len(tup):
        raise ValueError(f"Expected tuple of length {length}, got input {tup}")
    return tup
