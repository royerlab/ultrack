from typing import List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from skimage.measure import regionprops_table
from toolz import curry

from ultrack.utils.array import check_array_chunk
from ultrack.utils.multiprocessing import multiprocessing_apply


def intensity_sum(mask: np.ndarray, intensities: np.ndarray) -> float:
    """Sums the intensity inside the mask"""
    return intensities[mask].sum(axis=0, dtype=float)


def intensity_std(mask: np.ndarray, intensities: np.ndarray) -> float:
    """Standard deviation of the intensity inside the mask"""
    return intensities[mask].std(axis=0, dtype=float)


def num_pixels(mask: np.ndarray) -> int:
    """Number of pixels in the mask.
    NOTE: should be removed in the future, this was a bug in scikit-image
          https://github.com/scikit-image/scikit-image/issues/7038
    """
    return mask.sum()


@curry
def _extract_measurements(
    t: int,
    segments: ArrayLike,
    image: Optional[Union[List[ArrayLike], ArrayLike]],
    channel_axis: Optional[int],
    properties: Tuple[str, ...],
    extra_properties: Optional[List],
    spatial_scale: Optional[ArrayLike],
) -> pd.DataFrame:
    """
    Extract measurements for a single time point.
    See `tracks_properties` for parameter descriptions.
    """
    if image is not None:
        # create image with channel as the last dimension
        if isinstance(image, list):
            t_image = np.stack([np.asarray(channel[t]) for channel in image], axis=-1)
        else:
            t_image = np.asarray(image[t])
            if channel_axis is not None:
                axes = list(range(t_image.ndim))
                axes.remove(channel_axis)
                axes.append(channel_axis)
                t_image = t_image.transpose(axes)
    else:
        t_image = None

    df = pd.DataFrame(
        regionprops_table(
            np.asarray(segments[t]),
            intensity_image=t_image,
            properties=properties,
            separator="_",
            extra_properties=extra_properties,
            spacing=spatial_scale,
        ),
    )

    df["t"] = t

    return df


def tracks_properties(
    segments: ArrayLike,
    tracks_df: Optional[pd.DataFrame] = None,
    image: Optional[Union[List[ArrayLike], ArrayLike]] = None,
    properties: Tuple[str, ...] = (
        "label",
        "num_pixels",
        "area",
        "intensity_sum",
        "intensity_mean",
        "intensity_std",
        "intensity_min",
        "intensity_max",
    ),
    channel_axis: Optional[int] = None,
    scale: Optional[ArrayLike] = None,
    n_workers: int = 1,
) -> pd.DataFrame:
    """
    Calculate properties of tracked regions from segmentation data.

    Parameters
    ----------
    segments : ArrayLike
        Array-like object containing the segmented regions for each time point.
        Time must be the first dimension.
    tracks_df : Optional[pd.DataFrame], default None
        DataFrame containing tracking information for regions.
        When provided, measurements are merged into this data frame.
    image : Optional[Union[List[ArrayLike], ArrayLike]], default None
        Array-like object containing the image data for each time point. If provided,
        intensity-based properties will be calculated; otherwise, only geometric
        properties will be calculated.
    properties : Tuple[str, ...], default ('num_pixels', 'area', 'intensity_sum',
                                          'intensity_mean', 'intensity_std', 'intensity_min',
                                          'intensity_max')
        Tuple of property names to be calculated. Available options include:
            - 'num_pixels': Number of pixels in the region.
            - 'area': Area of the region.
            - 'intensity_sum': Sum of pixel intensities within the region.
            - 'intensity_mean': Mean pixel intensity within the region.
            - 'intensity_std': Standard deviation of pixel intensities within the region.
            - 'intensity_min': Minimum pixel intensity within the region.
            - 'intensity_max': Maximum pixel intensity within the region.
    channel_axis : Optional[int], default None
        If the `image` parameter is provided and it contains multi-channel data, this
        parameter specifies the axis containing the channels. If None, the data is assumed
        to be single-channel.
    scale : Optional[ArrayLike], default None
        Array-like object containing the scaling factors for each dimension of segments.
        Must include time dimension.
        Used for converting pixel measurements to physical units.
    n_workers : int, default 1
        Number of workers to use for parallel processing.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the calculated properties for each region at each time point.

    Notes
    -----
    - The function uses regionprops_table from skimage.measure to calculate properties.
    - If `image` is None, only geometric properties will be calculated.
    - If `image` is provided, intensity-based properties will be calculated, provided they
      are included in the `properties` parameter.

    Examples
    --------
    >>> # Example usage with image data
    >>> image_data = ...  # Provide image data here
    >>> segments_data = ...  # Provide segmentation data here
    >>> result_df = tracks_properties(segments_data, image=image_data)

    >>> # Example usage without image data
    >>> segments_data = ...  # Provide segmentation data here
    >>> result_df = tracks_properties(segments_data)
    """
    if scale is None:
        spatial_scale = None
    else:
        # excluding non-spatial dimensions
        spatial_scale = scale[-segments.ndim + 1 :]

    rename_map = {"label": "track_id"}
    extra_properties = []

    if "num_pixels" in properties:
        extra_properties.append(num_pixels)

    if image is None:
        properties = tuple(p for p in properties if "intensity" not in p)

    else:
        if not isinstance(image, list):
            if image.ndim > segments.ndim and channel_axis is None:
                raise ValueError(
                    "Channel axis must be specified when multi-channel image is provided."
                )
            check_array_chunk(image)
        else:
            for channel in image:
                check_array_chunk(channel)

        if "intensity_sum" in properties:
            extra_properties.append(intensity_sum)

        if "intensity_std" in properties:
            extra_properties.append(intensity_std)

    extract_measurements = _extract_measurements(
        segments=segments,
        image=image,
        channel_axis=channel_axis,
        properties=properties,
        extra_properties=extra_properties,
        spatial_scale=spatial_scale,
    )

    measures = multiprocessing_apply(
        extract_measurements,
        range(segments.shape[0]),
        n_workers=n_workers,
        desc="Extracting measurements",
    )

    measures = pd.concat(measures)
    measures.rename(columns=rename_map, inplace=True)
    measures.reset_index(inplace=True, drop=True)

    if tracks_df is not None:
        measures = tracks_df.merge(
            measures,
            on=["t", "track_id"],
            how="left",
        )

    return measures
