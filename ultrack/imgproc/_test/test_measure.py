from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import zarr

from ultrack.imgproc.measure import tracks_properties


@pytest.fixture
def mock_tracks_df(num_tracks=10, num_timepoints=2):
    data = {
        "track_id": np.repeat(np.arange(1, num_tracks + 1), num_timepoints),
        "t": np.tile(np.arange(num_timepoints), num_tracks),
    }
    return pd.DataFrame(data)


@pytest.mark.parametrize(
    "timelapse_mock_data",
    [
        {"length": 2, "size": 64, "n_dim": 2},
        {"length": 2, "size": 32, "n_dim": 3},
    ],
    indirect=True,
)
def test_tracks_properties_geometric(
    timelapse_mock_data: Tuple[zarr.Array, zarr.Array, zarr.Array],
):
    _, _, segmentation = timelapse_mock_data
    result_df = tracks_properties(
        segments=segmentation,
        n_workers=2,
    )

    # Ensure DataFrame contains the expected columns
    expected_columns = ["track_id", "t", "num_pixels", "area"]
    assert all(column in result_df.columns for column in expected_columns)

    # Ensure intensity-related columns are not present
    assert "intensity_sum" not in result_df.columns
    assert "intensity_mean" not in result_df.columns
    assert "intensity_std" not in result_df.columns
    assert "intensity_min" not in result_df.columns
    assert "intensity_max" not in result_df.columns

    # Ensure the DataFrame has the correct number of rows
    # subtracking background
    assert len(result_df) == sum(np.unique(s).shape[0] - 1 for s in segmentation)


@pytest.mark.parametrize(
    "timelapse_mock_data",
    [
        {"length": 2, "size": 32, "n_dim": 3},
    ],
    indirect=True,
)
def test_tracks_properties_merge(
    timelapse_mock_data: Tuple[zarr.Array, zarr.Array, zarr.Array],
    mock_tracks_df: pd.DataFrame,
) -> None:
    # Test case for merging with existing tracks DataFrame
    _, contours, segmentation = timelapse_mock_data
    result_df = tracks_properties(
        segments=segmentation,
        image=contours,
        tracks_df=mock_tracks_df,
    )

    # Ensure DataFrame contains the expected columns
    expected_columns = [
        "track_id",
        "t",
        "num_pixels",
        "area",
        "intensity_sum",
        "intensity_mean",
        "intensity_std",
        "intensity_min",
        "intensity_max",
    ]
    assert all(column in result_df.columns for column in expected_columns)

    # Ensure the DataFrame has the correct number of rows
    assert len(result_df) == len(mock_tracks_df)
