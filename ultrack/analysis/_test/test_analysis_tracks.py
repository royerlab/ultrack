import pandas as pd

from ultrack.analysis.tracks import displacement


def test_tracks_displacement():
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
    result = displacement(df)

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
