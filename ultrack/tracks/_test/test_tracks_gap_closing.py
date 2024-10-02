import numpy as np
import pandas as pd
import zarr
import zarr.storage

from ultrack.tracks import close_tracks_gaps
from ultrack.utils.constants import NO_PARENT


def test_gap_closing() -> None:
    """

    track id's shown by numbers

    8      ---

    2   --  -7 -9
       /
    1-       - 5
       \\    /
    3   -- -   4
            \
             - 6
    """

    segments_with_gap = np.zeros((8, 7), dtype=np.uint8)

    # Create a DataFrame with tracks
    tracks_df_with_gap = pd.DataFrame(
        {
            "track_id": [1, 2, 2, 3, 3, 4, 5, 6, 7, 8, 8, 8, 9],
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
            "t": [0, 1, 2, 1, 2, 4, 5, 5, 5, 4, 5, 6, 7.0],
            "x": [2, 0, 0, 4, 4, 4, 5, 6, 1, 3, 3, 3, 2.0],
        }
    )

    segments_with_gap[
        tracks_df_with_gap["t"].astype(int), tracks_df_with_gap["x"].astype(int)
    ] = tracks_df_with_gap["track_id"].values
    segments_with_gap = zarr.array(segments_with_gap)

    tracks_df, segments = close_tracks_gaps(
        tracks_df_with_gap,
        max_gap=2,
        max_radius=1.5,
        spatial_columns=["x"],
        segments=segments_with_gap,
    )
    expected_tracks_df = pd.DataFrame(
        {
            "track_id": [1, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 5, 6, 8, 8, 8],
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
                1,
                3,
                3,
                NO_PARENT,
                NO_PARENT,
                NO_PARENT,
            ],
            "t": [0, 1, 2, 3, 4, 5, 6, 7, 1, 2, 3, 4, 5, 5, 4, 5, 6.0],
            "x": [2, 0, 0, 1 / 3, 2 / 3, 1, 1.5, 2, 4, 4, 4, 4, 5, 6, 3, 3, 3],
        }
    )

    expected_segments = np.zeros((8, 7), dtype=np.uint8)
    expected_segments[
        expected_tracks_df["t"].astype(int),
        np.floor(expected_tracks_df["x"]).astype(int),
    ] = expected_tracks_df["track_id"].values

    # +1 of 9 to 7
    # +2 of 7 to 2
    # +1 of 4 to 3
    assert tracks_df.shape[0] == tracks_df_with_gap.shape[0] + 1 + 2 + 1

    # forcing equal index
    expected_tracks_df.index = tracks_df.index

    pd.testing.assert_frame_equal(tracks_df, expected_tracks_df, check_like=True)

    np.testing.assert_array_equal(segments, expected_segments)
