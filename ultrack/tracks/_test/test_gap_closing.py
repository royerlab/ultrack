import pandas as pd

from ultrack.core.database import NO_PARENT
from ultrack.tracks import close_tracks_gaps


def test_gap_closing() -> None:
    """

    track id's shown by numbers

    8      ---

    2   --  -- 7
       /
    1-       - 5
       \\    /
    3   -- -   4
            \
             - 6
    """

    # Create a DataFrame with tracks
    tracks_df_with_gap = pd.DataFrame(
        {
            "track_id": [1, 2, 2, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],
            "parent_track_id": [
                NO_PARENT,
                1,
                1,
                1,
                1,
                NO_PARENT,
                4,
                4,
                NO_PARENT,
                NO_PARENT,
                NO_PARENT,
                NO_PARENT,
                NO_PARENT,
            ],
            "t": [0, 1, 2, 1, 2, 4, 5, 5, 5, 6, 4, 5, 6.0],
            "x": [2, 0, 0, 4, 4, 4, 5, 6, 1, 1, 3, 3, 3.0],
        }
    )

    tracks_df = close_tracks_gaps(
        tracks_df_with_gap, max_gap=2, max_radius=1.5, spatial_columns=["x"]
    )
    expected_tracks_df = pd.DataFrame(
        {
            "track_id": [1, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 5, 6, 8, 8, 8],
            "parent_track_id": [
                NO_PARENT,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                1,
                3,
                3,
                NO_PARENT,
                NO_PARENT,
                NO_PARENT,
            ],
            "t": [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 5, 4, 5, 6.0],
            "x": [2, 0, 0, 1 / 3, 2 / 3, 1, 1, 4, 4, 4, 4, 5, 6, 3, 3, 3.0],
        }
    )
    assert tracks_df.shape[0] == tracks_df_with_gap.shape[0] + 1 + 2

    # forcing equal index
    expected_tracks_df.index = tracks_df.index

    pd.testing.assert_frame_equal(tracks_df, expected_tracks_df, check_like=True)
