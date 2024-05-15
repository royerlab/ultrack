from pathlib import Path
from typing import Optional, Sequence

import click
from napari.plugins import _initialize_plugins
from napari.viewer import ViewerModel

from ultrack.cli.utils import (
    napari_reader_option,
    output_directory_option,
    overwrite_option,
    paths_argument,
)
from ultrack.utils.data import validate_and_overwrite_path
from ultrack.utils.edge import labels_to_contours


@click.command("labels_to_contours")
@paths_argument()
@output_directory_option(help="`detection.zarr` and `contours.zarr` output directory.")
@napari_reader_option()
@click.option(
    "--sigma",
    "-s",
    type=float,
    default=None,
    show_default=True,
    help="Contour smoothing parameter (gaussian blur sigma). No blurring by default.",
)
@overwrite_option()
def labels_to_contours_cli(
    paths: Sequence[Path],
    output_directory: Path,
    reader_plugin: str,
    sigma: Optional[float],
    overwrite: bool,
) -> None:
    """
    Converts and merges a sequence of labels into ultrack input format (foreground and contours)
    """
    foreground_path = output_directory / "foreground.zarr"
    validate_and_overwrite_path(foreground_path, overwrite, "cli")

    contours_path = output_directory / "contours.zarr"
    validate_and_overwrite_path(contours_path, overwrite, "cli")

    _initialize_plugins()

    viewer = ViewerModel()
    viewer.open(path=paths, plugin=reader_plugin)

    labels = [layer.data for layer in viewer.layers]
    del viewer

    labels_to_contours(
        labels,
        sigma=sigma,
        foreground_store_or_path=foreground_path,
        contours_store_or_path=contours_path,
    )
