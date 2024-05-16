from pathlib import Path
from typing import Optional, Sequence

import click
from napari.plugins import _initialize_plugins
from napari.viewer import ViewerModel
from numpy.typing import ArrayLike

from ultrack import segment
from ultrack.cli.utils import (
    batch_index_option,
    config_option,
    napari_reader_option,
    overwrite_option,
    paths_argument,
)
from ultrack.config import MainConfig


def _get_layer_data(viewer: ViewerModel, key: str) -> ArrayLike:
    """Get layer data from napari viewer."""
    layer = viewer.layers[key]
    if layer.multiscale:
        return layer.data[0]
    else:
        return layer.data


@click.command("segment")
@paths_argument()
@napari_reader_option()
@config_option()
@click.option(
    "--foreground-layer",
    "-fl",
    required=True,
    type=str,
    help="Cell foreground layer index on napari.",
)
@click.option(
    "--contours-layer",
    "-cl",
    required=True,
    type=str,
    help="Cell contours layer index on napari.",
)
@click.option(
    "--insertion-throttle-rate",
    required=False,
    type=int,
    help="Rate at which to insert new hierarchies (group of competing segments) into the database.",
    default=50,
)
@batch_index_option()
@overwrite_option()
def segmentation_cli(
    paths: Sequence[Path],
    reader_plugin: str,
    config: MainConfig,
    foreground_layer: str,
    contours_layer: str,
    insertion_throttle_rate: int,
    batch_index: Optional[int],
    overwrite: bool,
) -> None:
    """Compute candidate segments for tracking model from input data."""
    _initialize_plugins()

    viewer = ViewerModel()
    viewer.open(path=paths, plugin=reader_plugin)

    foreground = _get_layer_data(viewer, foreground_layer)
    edge = _get_layer_data(viewer, contours_layer)

    if batch_index is None or batch_index == 0:
        # this is not saved inside the `segment` function because this info
        # isn't available there
        config.data_config.metadata_add(
            {"scale": viewer.layers[contours_layer].scale.tolist()}
        )

    del viewer

    segment(
        foreground,
        edge,
        config,
        batch_index=batch_index,
        overwrite=overwrite,
        insertion_throttle_rate=insertion_throttle_rate,
    )
