from pathlib import Path
from typing import Optional, Tuple

import click
import zarr
from tifffile import imread

from ultrack.cli.utils import (
    config_option,
    output_directory_option,
    overwrite_option,
    tuple_callback,
)
from ultrack.config import MainConfig
from ultrack.core.export import to_ctc, to_tracks_layer, tracks_to_zarr
from ultrack.core.export.utils import maybe_overwrite_path


@click.command("ctc")
@output_directory_option(
    help="Output directory to save segmentation and lineage graph (e.g. 01_RES)."
)
@config_option()
@overwrite_option()
@click.option(
    "--margin",
    "-ma",
    default=0,
    type=int,
    show_default=True,
    help="Ignored margin on xy-plane.",
)
@click.option(
    "--scale",
    "-s",
    default=None,
    type=str,
    show_default=True,
    help="Output scale factor (e.g. 0.2,1,1 ). Useful when tracking was done on upscaled input."
    "Must have length 3, first dimension is ignored on for 2-d images.",
    callback=tuple_callback(length=3, dtype=float),
)
@click.option(
    "--first-frame-path",
    default=None,
    type=click.Path(path_type=Path, exists=True),
    show_default=True,
    help="Optional first frame path used to select a subset of lineages connected to this reference annotations.",
)
@click.option(
    "--dilation-iters",
    "-di",
    default=0,
    type=int,
    show_default=True,
    help="Iterations of radius 1 morphological dilations on labels, applied after scaling.",
)
@click.option(
    "--stitch-tracks",
    default=False,
    is_flag=True,
    type=bool,
    help="Stitches (connects) incomplete tracks nearby tracks on subsequent time point.",
)
def ctc_cli(
    output_directory: Path,
    config: MainConfig,
    margin: int,
    scale: Optional[Tuple[float]],
    first_frame_path: Optional[Path],
    dilation_iters: int,
    stitch_tracks: bool,
    overwrite: bool,
) -> None:
    """Exports tracking results to cell-tracking challenge (http://celltrackingchallenge.net) format."""

    if first_frame_path is None:
        first_frame = None
    else:
        first_frame = imread(first_frame_path)

    to_ctc(
        output_directory,
        config.data_config,
        margin=margin,
        scale=scale,
        first_frame=first_frame,
        dilation_iters=dilation_iters,
        stitch_tracks=stitch_tracks,
        overwrite=overwrite,
    )


@click.command("zarr-napari")
@output_directory_option(
    help="Output directory to save segmentation masks and tracks table (e.g. results)."
)
@config_option()
@overwrite_option()
def zarr_napari_cli(
    output_directory: Path,
    config: MainConfig,
    overwrite: bool,
) -> None:
    """
    Exports segments to zarr and tracks to napari tabular format
    (.csv for tracklets, parent relationship is lost).
    """
    tracks_path = output_directory / "tracks.csv"
    maybe_overwrite_path(tracks_path, overwrite)

    segm_path = output_directory / "segments.zarr"
    maybe_overwrite_path(segm_path, overwrite)

    output_directory.mkdir(exist_ok=True)

    tracks, _ = to_tracks_layer(config.data_config)
    tracks.to_csv(tracks_path, index=False)

    store = zarr.NestedDirectoryStore(segm_path)
    tracks_to_zarr(config.data_config, tracks, store=store)


@click.group("export")
def export_cli() -> None:
    """Exports tracking and segmentation results to selected format."""


export_cli.add_command(ctc_cli)
export_cli.add_command(zarr_napari_cli)
