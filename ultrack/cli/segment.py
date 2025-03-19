from pathlib import Path
from typing import Optional, Sequence

import click
import dask.array as da
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
    "--images-layer",
    "-il",
    required=False,
    multiple=True,
    type=str,
    help="Image layer index on napari for intensity features, it can used multiple times for multiple channels.",
)
@click.option(
    "--insertion-throttle-rate",
    required=False,
    type=int,
    help="Rate at which to insert new hierarchies (group of competing segments) into the database.",
    default=50,
)
@click.option(
    "--properties",
    "-p",
    type=str,
    multiple=True,
    help="Compute properties of the segments, it can be used multiple times for multiple properties.",
)
@batch_index_option()
@overwrite_option()
def segmentation_cli(
    paths: Sequence[Path],
    reader_plugin: str,
    config: MainConfig,
    foreground_layer: str,
    contours_layer: str,
    images_layer: Sequence[str],
    insertion_throttle_rate: int,
    properties: Sequence[str],
    batch_index: Optional[int],
    overwrite: bool,
) -> None:
    """Compute candidate segments for tracking model from input data."""
    _initialize_plugins()

    viewer = ViewerModel()
    viewer.open(path=paths, plugin=reader_plugin)

    foreground = _get_layer_data(viewer, foreground_layer)
    edge = _get_layer_data(viewer, contours_layer)

    if len(images_layer) == 0:
        images = None
    else:
        has_intensity = any("intensity" in prop for prop in properties)
        if not has_intensity:
            raise ValueError(
                "Intensity features are required to compute image intensity properties.\n"
                "Found properties: {}\n".format(properties),
                "Expected properties: intensity_mean, intensity_std, intensity_sum or intensity_min, intensity_max",
            )

        if len(images_layer) == 1:
            images = _get_layer_data(viewer, images_layer[0])
        else:
            images = da.stack(
                [_get_layer_data(viewer, key) for key in images_layer], axis=-1
            )

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
        images=images,
        properties=None if len(properties) == 0 else properties,
    )
