from pathlib import Path
from typing import Optional, Sequence

import click
from napari.plugins import _initialize_plugins
from napari.viewer import ViewerModel

from ultrack import link
from ultrack.cli.utils import (
    batch_index_option,
    channel_axis_option,
    config_option,
    napari_reader_option,
    overwrite_option,
    paths_argument,
)
from ultrack.config import MainConfig


@click.command("link")
@paths_argument()
@napari_reader_option()
@config_option()
@channel_axis_option(
    default=None,
    help="Channel axis, only used when input `paths` are provided",
)
@batch_index_option()
@overwrite_option()
def link_cli(
    paths: Sequence[Path],
    reader_plugin: str,
    config: MainConfig,
    channel_axis: Optional[int],
    batch_index: Optional[int],
    overwrite: bool,
) -> None:
    """Links segmentation candidates adjacent in time."""

    images = tuple()
    if len(paths) > 0:
        _initialize_plugins()

        viewer = ViewerModel()

        kwargs = {}
        if channel_axis is not None:
            kwargs["channel_axis"] = channel_axis

        images = [
            layer.data[0] if layer.multiscale else layer.data
            for layer in viewer.open(paths, **kwargs, plugin=reader_plugin)
        ]
        del viewer

    link(
        config,
        images=images,
        scale=config.data_config.metadata.get("scale"),
        batch_index=batch_index,
        overwrite=overwrite,
    )
