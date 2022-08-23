from pathlib import Path
from typing import Optional, Tuple

import click
from tifffile import imread

from ultrack.cli.utils import config_option, overwrite_option
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
    type=tuple,
    show_default=True,
    help="Output scale factor (e.g. (0.2, 1, 1)). Useful when tracking was done on upscaled input.",
)
@click.option(
    "--first-frame-path",
    default=None,
    type=click.Path(path_type=Path, exists=True),
    show_default=True,
    help="Optional first frame path used to select a subset of lineages connected to this reference annotations.",
)
def ctc_cli(
    output_directory: Path,
    config: MainConfig,
    scale: Optional[Tuple[float]],
    first_frame_path: Optional[Path],
    overwrite: bool,
) -> None:
    """Exports tracking results to cell-tracking challenge (http://celltrackingchallenge.net) format."""

    if first_frame_path is None:
        first_frame = None
    else:
        first_frame = imread(first_frame_path)

    to_ctc(output_directory, config.data_config, scale, first_frame, overwrite)


@click.group("export")
def export_cli() -> None:
    """Exports tracking and segmentation results to selected format."""


export_cli.add_command(ctc_cli)
