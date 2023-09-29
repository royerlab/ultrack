from pathlib import Path
from typing import Optional, Sequence

import click
from napari.plugins import _initialize_plugins
from napari.viewer import ViewerModel

from ultrack.cli.utils import (
    channel_axis_option,
    config_option,
    napari_reader_option,
    paths_argument,
)
from ultrack.config import MainConfig
from ultrack.imgproc.flow import add_flow


@click.command("add_flow")
@paths_argument()
@napari_reader_option()
@config_option()
@channel_axis_option(default=0, help="Coordinates shift axis.")
def add_flow_cli(
    paths: Sequence[Path],
    reader_plugin: str,
    config: MainConfig,
    channel_axis: Optional[int],
) -> None:
    """Adds coordinates shift (vector field) to segmentation hypotheses."""
    _initialize_plugins()

    viewer = ViewerModel()

    vector_field = [
        layer.data[0] if layer.multiscale else layer.data
        for layer in viewer.open(paths, channel_axis=channel_axis, plugin=reader_plugin)
    ]
    del viewer

    add_flow(
        config,
        vector_field=vector_field,
    )
