import numpy as np
import pandas as pd
import pytest

from ultrack.analysis import tracks_profile_matrix


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
