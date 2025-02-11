from typing import Sequence

import numpy as np
import pandas as pd
import pytest

from ultrack.tracks.stats import (
    estimate_drift,
    tracks_df_movement,
    tracks_length,
    tracks_profile_matrix,
)
from ultrack.utils.constants import NO_PARENT


def spatial_df(group_drift: Sequence[int], length_per_group: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    df = []
    for d in group_drift:
        pos = np.empty((length_per_group, 4))
        pos[:, 0] = np.arange(length_per_group)
        pos[0, 1:] = rng.uniform(size=3)

        for i in range(length_per_group - 1):
            step = rng.uniform(size=3)
            pos[i + 1, 1:] = pos[i, 1:] + d * step / np.linalg.norm(step)
        df.append(pd.DataFrame(pos, columns=["t", "z", "y", "x"]))
    df = pd.concat(df)
    return df


@pytest.mark.parametrize(
    "drift",
    [0.5, 5, 10],
)
def test_spatial_drift(drift: float) -> None:
    df = spatial_df([drift])
    estimated_drift = np.linalg.norm(tracks_df_movement(df, lag=1), axis=1)
    assert np.allclose(estimated_drift[1:], drift)
    assert np.allclose(estimated_drift[:1], 0)


@pytest.mark.parametrize(
    "length_per_group",
    [3, 8, 15],
)
def test_maximum_distance_estimate(length_per_group: int) -> None:
    group_drift = [1, 10, 20]
    df = spatial_df(group_drift, length_per_group)
    df["track_id"] = np.repeat(np.arange(len(group_drift)), length_per_group)
    assert np.allclose(estimate_drift(df), max(group_drift))


def test_tracks_df_movement():
    # Sample test data
    df = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 2],
            "t": [1, 2, 1, 2],
            "z": [0, 1, 0, 2],
            "y": [1, 2, 1, 2],
            "x": [2, 3, 2, 2],
        }
    )

    # Call the function
    result = tracks_df_movement(df)

    # Expected result
    expected = pd.DataFrame(
        {
            "z": [0.0, 1.0, 0.0, 2.0],
            "y": [0.0, 1.0, 0.0, 1.0],
            "x": [0.0, 1.0, 0.0, 0.0],
        }
    )

    # Assert that the result matches the expected dataframe
    pd.testing.assert_frame_equal(result, expected)


def test_tracks_df_movement_2d():
    # Sample test data
    df = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 2],
            "t": [1, 2, 1, 2],
            "y": [1, 2, 1, 2],
            "x": [2, 3, 2, 2],
        }
    )

    # Call the function
    result = tracks_df_movement(df)

    # Expected result
    expected = pd.DataFrame(
        {
            "y": [0.0, 1.0, 0.0, 1.0],
            "x": [0.0, 1.0, 0.0, 0.0],
        }
    )

    # Assert that the result matches the expected dataframe
    pd.testing.assert_frame_equal(result, expected)


def test_tracks_profile_matrix_one_track_one_attribute():
    tracks_df = pd.DataFrame(
        {"track_id": [1, 1, 1], "t": [0, 1, 2], "attribute_1": [10, 20, 30]}
    )
    columns = ["attribute_1"]
    result = tracks_profile_matrix(tracks_df, columns)
    expected_result = np.array([[10, 20, 30]])
    assert np.array_equal(result, expected_result)


def test_tracks_profile_matrix_multiple_tracks_multiple_attributes():
    tracks_df = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 2, 2],
            "t": [0, 1, 0, 1, 2],
            "attribute_1": [10, 20, 30, 40, 50],
            "attribute_2": [100, 200, 300, 400, 500],
        }
    )
    columns = ["attribute_1", "attribute_2"]
    result = tracks_profile_matrix(tracks_df, columns)
    expected_result = np.array(
        [[[10, 20, 0], [30, 40, 50]], [[100, 200, 0], [300, 400, 500]]]
    )
    assert np.array_equal(result, expected_result)


def test_tracks_profile_matrix_missing_timesteps():
    tracks_df = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 2, 2],
            "t": [0, 1, 0, 2, 3],
            "attribute_1": [10, 20, 30, 40, 50],
            "attribute_2": [100, 200, 300, 400, 500],
        }
    )
    columns = ["attribute_1", "attribute_2"]
    result = tracks_profile_matrix(tracks_df, columns)
    expected_result = np.array(
        [[[10, 20, 0, 0], [30, 0, 40, 50]], [[100, 200, 0, 0], [300, 0, 400, 500]]]
    )
    assert np.array_equal(result, expected_result)


def test_tracks_profile_matrix_missing_attributes():
    tracks_df = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 2, 2],
            "t": [0, 1, 0, 1, 2],
            "attribute_1": [10, 20, 30, 40, 50],
        }
    )
    columns = ["attribute_1", "attribute_2"]
    with pytest.raises(ValueError):
        tracks_profile_matrix(tracks_df, columns)


def test_tracks_length():
    # Sample data
    data = {
        "t": [0, 1, 2, 3, 4, 3, 4, 5, 2, 3, 4, 5, 4, 5],
        "track_id": [1, 1, 1, 2, 2, 3, 3, 3, 4, 4, 5, 5, 6, 6],
        "parent_track_id": [
            NO_PARENT,
            NO_PARENT,
            NO_PARENT,
            1,
            1,
            1,
            1,
            1,
            NO_PARENT,
            NO_PARENT,
            4,
            4,
            4,
            4,
        ],
    }
    df = pd.DataFrame(data)

    length_df = tracks_length(df)
    np.testing.assert_array_equal(length_df["length"], [3, 2, 3, 2, 2, 2])
    np.testing.assert_array_equal(length_df["start"], [0, 3, 3, 2, 4, 4])
    np.testing.assert_array_equal(length_df["end"], [2, 4, 5, 3, 5, 5])

    length_df = tracks_length(df, include_appearing=False)
    assert 4 not in length_df["track_id"].to_numpy()
    np.testing.assert_array_equal(length_df["length"], [3, 2, 3, 2, 2])
    np.testing.assert_array_equal(length_df["start"], [0, 3, 3, 4, 4])
    np.testing.assert_array_equal(length_df["end"], [2, 4, 5, 5, 5])

    length_df = tracks_length(df, include_disappearing=False)
    assert 2 not in length_df["track_id"].to_numpy()
    np.testing.assert_array_equal(length_df["length"], [3, 3, 2, 2, 2])
    np.testing.assert_array_equal(length_df["start"], [0, 3, 2, 4, 4])
    np.testing.assert_array_equal(length_df["end"], [2, 5, 3, 5, 5])

    length_df = tracks_length(df, include_disappearing=False, include_appearing=False)
    assert 2 not in length_df["track_id"].to_numpy()
    assert 4 not in length_df["track_id"].to_numpy()
    np.testing.assert_array_equal(length_df["length"], [3, 3, 2, 2])
    np.testing.assert_array_equal(length_df["start"], [0, 3, 4, 4])
    np.testing.assert_array_equal(length_df["end"], [2, 5, 5, 5])
