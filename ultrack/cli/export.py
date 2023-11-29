from pathlib import Path
from typing import Optional, Tuple

import click
from napari.plugins import _initialize_plugins
from napari.viewer import ViewerModel
from tifffile import imread
from tqdm import tqdm

from ultrack.cli.utils import (
    config_option,
    napari_reader_option,
    output_directory_option,
    overwrite_option,
    tuple_callback,
)
from ultrack.config import MainConfig
from ultrack.core.export import to_ctc, to_trackmate, to_tracks_layer, tracks_to_zarr
from ultrack.core.solve.sqltracking import SQLTracking
from ultrack.imgproc.measure import tracks_properties
from ultrack.utils.data import validate_and_overwrite_path


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
        config,
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
@click.option(
    "--measure",
    default=False,
    show_default=True,
    is_flag=True,
    help="Add segmentation measurements to tracks table.",
)
@click.option(
    "--image-path",
    "-i",
    type=click.Path(path_type=Path, exists=True),
    default=None,
    help="If provided, tracks measurements will be enriched with intensity-based properties.",
)
@napari_reader_option()
def zarr_napari_cli(
    output_directory: Path,
    config: MainConfig,
    overwrite: bool,
    measure: bool,
    image_path: Optional[Path],
    reader_plugin: str,
) -> None:
    """
    Exports segments to zarr and tracks to napari tabular format
    (.csv for tracklets, parent relationship is lost).
    """
    tracks_path = output_directory / "tracks.csv"
    validate_and_overwrite_path(tracks_path, overwrite, "cli")

    segm_path = output_directory / "segments.zarr"
    validate_and_overwrite_path(segm_path, overwrite, "cli")

    output_directory.mkdir(exist_ok=True)

    tracks, _ = to_tracks_layer(config, include_parents=True)
    tracks.to_csv(tracks_path, index=False)  # saving before measures just to be sure

    segments = tracks_to_zarr(config, tracks, store_or_path=segm_path)

    if measure or image_path is not None:
        # extract segmentation measurements
        if image_path is None:
            image = None
        else:
            _initialize_plugins()
            viewer = ViewerModel()
            image = [
                layer.data[0] if layer.multiscale else layer.data
                for layer in viewer.open(image_path, plugin=reader_plugin)
            ]
            del viewer

        tracks_w_measures = tracks_properties(
            segments=segments,
            tracks_df=tracks,
            image=image,
            scale=config.data_config.metadata.get("scale"),
            n_workers=config.data_config.n_workers,
        )
        tracks_w_measures.to_csv(tracks_path, index=False)


@click.command("trackmate")
@click.option(
    "--output-path",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    show_default=True,
    help="TrackMate XML output path.",
)
@config_option()
@overwrite_option()
def trackmate_cli(
    config: MainConfig,
    output_path: Path,
    overwrite: bool,
) -> None:
    """
    Exports tracking results to TrackMate XML format.
    """
    to_trackmate(config, output_path, overwrite)


@click.command("lp")
@click.option(
    "--output-path",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    show_default=True,
    help="ILP model (.lp) output path.",
)
@config_option()
@overwrite_option()
def lp_cli(
    config: MainConfig,
    output_path: Path,
    overwrite: bool,
) -> None:
    """
    Exports tracking ILP to .lp format.
    """
    tracker = SQLTracking(config)

    for batch_index in tqdm(range(tracker.num_batches), "Exporting ILP"):
        tracker.construct_model(index=batch_index)

        if tracker.num_batches == 1:
            model_path = output_path
        else:
            name = f"{output_path.name.removesuffix('.lp')}_{batch_index:02d}.lp"
            model_path = output_path.parent / name

        if model_path.exists() and not overwrite:
            raise ValueError(f"File {model_path} already exists. Set `--overwrite`.")

        tracker.solver._model.write(str(model_path))


@click.group("export")
def export_cli() -> None:
    """Exports tracking and segmentation results to selected format."""


export_cli.add_command(ctc_cli)
export_cli.add_command(lp_cli)
export_cli.add_command(trackmate_cli)
export_cli.add_command(zarr_napari_cli)
