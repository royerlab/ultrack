from pathlib import Path
from typing import Optional, Sequence
import pprint
import glob
import os
import re

import click
import numpy as np
from napari.plugins import _initialize_plugins
from napari.viewer import ViewerModel
import zarr

from ultrack import segment, link, solve
from ultrack.utils.array import array_apply
from ultrack.cli.utils import (
    napari_reader_option,
    overwrite_option,
)
from ultrack.config import MainConfig
from ultrack.core.export.geff import to_geff
from ultrack.core.export.trackmate import to_trackmate
from ultrack.utils.edge import labels_to_contours
from ultrack.imgproc.segmentation import detect_foreground
from ultrack.imgproc.intensity import robust_invert


def _get_layer_data(viewer: ViewerModel, key: str | int):
    """Get layer data from napari viewer."""
    layer = viewer.layers[key]
    if layer.multiscale:
        return layer.data[0]
    else:
        return layer.data


def _apply_config_overrides(config: MainConfig, overrides: Sequence[str]) -> MainConfig:
    """Apply parameter overrides to config.
    
    Parameters
    ----------
    config : MainConfig
        Base configuration to modify
    overrides : Sequence[str]
        Parameter overrides in format "section.parameter=value"
        
    Returns
    -------
    MainConfig
        Modified configuration
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: {override}. Expected format: 'section.parameter=value'")
        
        param_path, value_str = override.split("=", 1)
        sections = param_path.split(".")
        
        if len(sections) != 2:
            raise ValueError(f"Invalid parameter path: {param_path}. Expected format: 'section.parameter'")
            
        section_name, param_name = sections
        
        # Get the appropriate config section
        if section_name == "segmentation":
            config_section = config.segmentation_config
        elif section_name == "linking":
            config_section = config.linking_config
        elif section_name == "tracking":
            config_section = config.tracking_config
        elif section_name == "data":
            config_section = config.data_config
        else:
            raise ValueError(f"Unknown config section: {section_name}. Valid sections: segmentation, linking, tracking, data")
        
        # Check if parameter exists
        if not hasattr(config_section, param_name):
            raise ValueError(f"Parameter '{param_name}' not found in section '{section_name}'")
        
        # Get the parameter's current type and convert the string value
        current_value = getattr(config_section, param_name)
        if current_value is None:
            # If current value is None, try to infer type from string
            try:
                # Try int first
                new_value = int(value_str)
            except ValueError:
                try:
                    # Try float
                    new_value = float(value_str)
                except ValueError:
                    # Try boolean
                    if value_str.lower() in ('true', 'false'):
                        new_value = value_str.lower() == 'true'
                    else:
                        # Keep as string
                        new_value = value_str
        else:
            # Convert to the same type as current value
            current_type = type(current_value)
            if current_type == bool:
                new_value = value_str.lower() == 'true'
            elif current_type == int:
                new_value = int(value_str)
            elif current_type == float:
                new_value = float(value_str)
            else:
                new_value = value_str
        
        # Set the new value
        setattr(config_section, param_name, new_value)
        
    return config


def _preprocess_data(data, data_type: str, sigma: float, voxel_size: tuple[float, ...] | None = None):
    """Preprocess data based on type.
    
    Parameters
    ----------
    data : ArrayLike
        Input data array
    data_type : str
        Type of data: 'labels' or 'raw'
    sigma : Optional[float]
        Sigma parameter for smoothing
        
    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        Foreground and contours arrays
    """
    if voxel_size is not None:
        sigma = len(data.shape) * sigma

    if data_type == "labels":
        print("Converting labels to foreground and contours...")
        foreground, contours = array_apply(
            data,
            func=labels_to_contours,
            sigma=None,
            overwrite=True,
        )
        return foreground, contours
    
    elif data_type == "raw":
        print("Detecting foreground and generating contours from raw image...")
        foreground = array_apply(
            data,
            func=detect_foreground,
            sigma=sigma,
            remove_hist_mode=False,
            min_foreground=0.0,
            channel_axis=None,
        )
        contours = array_apply(
            data,
            func=robust_invert,
            sigma=1.0,
            lower_quantile=None,
            upper_quantile=0.9999,
        )
        
        return foreground, contours
    
    else:
        raise ValueError(f"Unknown data type: {data_type}. Expected 'labels' or 'raw'")


def _is_zarr_directory(path: Path) -> bool:
    """Check if a path is a valid zarr directory."""
    try:
        zarr.open(path, mode="r")
        return True
    except Exception:
        return False

def _get_data(paths: Sequence[Path], data_type: str, sigma: float, reader_plugin: str) -> tuple[np.ndarray, np.ndarray]:
        # Initialize napari plugins and load data
    _initialize_plugins()
    viewer = ViewerModel()

    stack = False
    if reader_plugin == "napari":
        if len(paths) > 1:
            stack = True
        elif len(paths) == 1:
            if Path(paths[0]).is_dir():
                if not _is_zarr_directory(paths[0]):
                    # Create a glob pattern to get all files
                    all_files = glob.glob(os.path.join(paths[0], '*'))

                    # Define regex pattern for image files
                    pattern = re.compile(r'.*\.(tif|tiff|png|jpg|jpeg|zarr)$', re.IGNORECASE)

                    # Filter and sort files
                    matched_files = sorted(f for f in all_files if pattern.match(f))
                    paths = [Path(f) for f in matched_files]
                    stack = True

    viewer.open(path=paths, plugin=reader_plugin, stack=stack)

    # Get the first layer's data (assuming single data source)
    if len(viewer.layers) == 0:
        raise ValueError("No data layers found in the provided paths.")
    
    # Use the first layer as the main data
    main_data = _get_layer_data(viewer, viewer.layers[0].name)
    
    # Preprocess data based on type
    foreground, contours = _preprocess_data(main_data, data_type, sigma)

    del viewer

    return foreground, contours


@click.command("trackmate", hidden=True, context_settings=dict(ignore_unknown_options=True))
@click.argument('path', nargs=1, type=click.Path(path_type=Path))
@click.option(
    "--output-path",
    "-o",
    type=Path,
    help="Path to save the output.",
)
@napari_reader_option()
# @click.option(
#     "--data-type",
#     "-dt",
#     required=True,
#     type=click.Choice(['labels', 'raw']),
#     help="Type of input data: 'labels' for label maps/segmentations or 'raw' for raw images.",
# )
@click.option(
    "--sigma",
    "-s",
    type=float,
    default=15.0,
    help="Sigma parameter for smoothing (labels) or contour detection (raw images).",
)
@click.argument('args', nargs=-1, type=click.UNPROCESSED)
@overwrite_option()
def trackmate_cli(
    path: Path,
    output_path: Path,
    reader_plugin: str,
    #data_type: str,
    sigma: float,
    args: Sequence[str],
    overwrite: bool,
) -> None:
    """Perform complete tracking pipeline (segmentation, linking, and solving) with default config.
    
    Accepts a directory containing data files (TIFF directory with all TIFFs to stack them, 
    single TIFF, zarr, etc.) and preprocesses the data based on the specified data type.
    """
    
    data_type = "raw"

    # Create default config
    config = MainConfig()
    
    # Apply parameter overrides
    config = _apply_config_overrides(config, args)
    pprint.pprint(config)

    # Get data
    paths = [path]
    foreground, contours = _get_data(paths, data_type, sigma, reader_plugin)

    # Run segmentation
    print("Running segmentation...")
    segment(
        foreground,
        contours,
        config,
        overwrite=overwrite,
    )

    # Run linking
    print("Running linking...")
    link(
        config,
        overwrite=overwrite,
    )

    # Run solving
    print("Running solving...")
    solve(config, overwrite=overwrite)
    
    print("Trackmate pipeline completed successfully!")

    try:
        to_geff(config, output_path, overwrite=overwrite)
    except Exception as e:
        print(f"Error exporting to GEFF: {e}, fallback to saving as TrackMate XML")
        to_trackmate(config, output_path.with_suffix(".xml"), overwrite=overwrite)


