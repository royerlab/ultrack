from pathlib import Path
from typing import Optional, Sequence

import click
import zarr
from napari.viewer import ViewerModel

from ultrack.cli.utils import (
    napari_reader_option,
    output_directory_option,
    overwrite_option,
)
from ultrack.core.export.utils import maybe_overwrite_path
from ultrack.utils.edge import labels_to_edges


@click.command("labels_to_edges")
@click.argument("paths", nargs=-1, type=click.Path(path_type=Path))
@output_directory_option(help="`detection.zarr` and `edges.zarr` output directory.")
@napari_reader_option()
@click.option(
    "--sigma",
    "-s",
    type=float,
    default=None,
    show_default=True,
    help="Edge smoothing parameter (gaussian blur sigma). No blurring by default.",
)
@overwrite_option()
def labels_to_edges_cli(
    paths: Sequence[Path],
    output_directory: Path,
    reader_plugin: str,
    sigma: Optional[float],
    overwrite: bool,
) -> None:
    """
    Converts and merges a sequence of labels into ultrack input format (detection and edges)
    """
    detection_path = output_directory / "detection.zarr"
    maybe_overwrite_path(detection_path, overwrite)

    edges_path = output_directory / "edges.zarr"
    maybe_overwrite_path(edges_path, overwrite)

    viewer = ViewerModel()
    viewer.open(path=paths, plugin=reader_plugin)

    labels_to_edges(
        [layer.data for layer in viewer.layers],
        sigma=sigma,
        detection_store=zarr.DirectoryStore(detection_path),
        edges_store=zarr.DirectoryStore(edges_path),
    )
