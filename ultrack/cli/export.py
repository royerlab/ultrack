from pathlib import Path
from typing import Optional, Tuple

import click
from tifffile import imread

from ultrack.cli.utils import config_option, overwrite_option, tuple_callback
from ultrack.config import MainConfig
from ultrack.core.export.ctc import to_ctc


@click.command("ctc")
@click.option(
    "--output-directory",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory to save segmentation masks and lineage graph (e.g. 01_RES).",
)
@config_option()
@overwrite_option()
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
    "--stitch-tracks",
    default=False,
    is_flag=True,
    type=bool,
    help="Stitches (connects) incomplete tracks nearby tracks on subsequent time point.",
)
def ctc_cli(
    output_directory: Path,
    config: MainConfig,
    scale: Optional[Tuple[float]],
    first_frame_path: Optional[Path],
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
        scale,
        first_frame,
        stitch_tracks,
        overwrite,
    )


@click.group("export")
def export_cli() -> None:
    """Exports tracking and segmentation results to selected format."""


export_cli.add_command(ctc_cli)
