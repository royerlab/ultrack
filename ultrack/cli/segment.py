from pathlib import Path
from typing import Sequence, Union

import click
from napari.viewer import ViewerModel

from ultrack import segment
from ultrack.cli.utils import config_option
from ultrack.config import MainConfig


@click.command("segment")
@click.argument("input-paths", nargs=-1, type=click.Path(path_type=Path), required=True)
@config_option()
def segmentation_cli(
    input_paths: Union[Sequence[Path], Path], config: MainConfig
) -> None:
    """Compute candidate segments for tracking model from input data."""

    viewer = ViewerModel()
    viewer.open(path=input_paths, plugin=config.reader_config.reader_plugin)

    detection = viewer.layers[config.reader_config.layer_indices[0]].data
    edge = viewer.layers[config.reader_config.layer_indices[1]].data

    segment(detection, edge, config.segmentation_config, config.data_config)
