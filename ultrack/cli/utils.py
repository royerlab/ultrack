import warnings
from pathlib import Path
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
            help="Ultrack configuration file (.toml)",
            callback=_config_callback,
        )(f)

    return decorator


def _batch_index_callback(
    ctx: click.Context, opt: click.Option, value: Optional[int]
) -> Optional[int]:
    if value is not None and ctx.params.get("overwrite", False):
        warnings.warn(
            "ATTENTION: Overwriting while using batch indices will result in multiple deletions."
        )
    return value


def batch_index_option() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--batch-index",
            "-b",
            required=False,
            default=None,
            show_default=True,
            type=int,
            callback=_batch_index_callback,
            help="Batch index to process a subset of time points. ATTENTION: this it not the time index.",
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
            help="Enables overwriting existing data.",
            is_eager=True,
        )(f)

    return decorator


def output_directory_option(**kwargs) -> Callable:

    if "required" not in kwargs and "default" not in kwargs:
        kwargs["required"] = True

    def decorator(f: Callable) -> Callable:
        return click.option(
            "--output-directory",
            "-o",
            type=click.Path(path_type=Path),
            **kwargs,
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


def napari_reader_option() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--reader-plugin",
            "-r",
            default="builtins",
            type=str,
            show_default=True,
            help="Napari reader plugin.",
        )(f)

    return decorator


def layer_key_option() -> Callable:
    def decorator(f: Callable) -> Callable:
        return click.option(
            "--layer-key",
            "-l",
            default=0,
            type=str,
            show_default=True,
            help="Layer key to index multi-channel input.",
        )(f)

    return decorator
