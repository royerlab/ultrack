import numpy as np
import pandas as pd
import pytest

from ultrack.core.export.ctc import ctc_compress_forest, stitch_tracks_df
from ultrack.tracks.graph import add_track_ids_to_tracks_df
from ultrack.utils.constants import NO_PARENT


@pytest.fixture
def dataframe_forest() -> pd.DataFrame:
    """
                  2
              --------
       1    /
    -------
            \\     3
              --------

            4
    ------------------
    """

    df = np.empty((40, 2), dtype=int)

    # track # 1
    df[:10, 0] = np.arange(1, 11)
    df[:10, 1] = df[:10, 0] - 1
    df[0, 1] = NO_PARENT

    # track # 2
    df[10:20, 0] = np.arange(11, 21)
    df[10:20, 1] = df[10:20, 0] - 1
    df[10, 1] = 10

    # track # 3
    df[20:30, 0] = np.arange(21, 31)
    df[20:30, 1] = df[20:30, 0] - 1
    df[20, 1] = 10

    df[30:40, 0] = np.arange(31, 41)
    df[30:40, 1] = df[30:40, 0] - 1
    df[30, 1] = NO_PARENT

    return pd.DataFrame(df[:, 1], index=df[:, 0], columns=["parent_id"])


@pytest.fixture
def dataframe_forest_with_time(dataframe_forest: pd.DataFrame) -> pd.DataFrame:
    df = dataframe_forest.copy()
    df["t"] = -1
    df["t"].values[:10] = np.arange(10)
    df["t"].values[10:20] = np.arange(10, 20)
    df["t"].values[20:30] = np.arange(10, 20)
    df["t"].values[30:40] = np.arange(5, 15)
    return df


def test_add_paths_to_forest(dataframe_forest: pd.DataFrame) -> None:
    df = add_track_ids_to_tracks_df(dataframe_forest)

    assert np.all(df["track_id"].values[0:10] == 1)
    assert np.all(df["track_id"].values[10:20] == 2)
    assert np.all(df["track_id"].values[20:30] == 3)
    assert np.all(df["track_id"].values[30:40] == 4)

    assert np.all(df["parent_track_id"].values[0:10] == NO_PARENT)
    assert np.all(df["parent_track_id"].values[10:30] == 1)
    assert np.all(df["parent_track_id"].values[30:40] == NO_PARENT)


def test_ctc_compress_forest(dataframe_forest_with_time: pd.DataFrame) -> pd.DataFrame:
    df = ctc_compress_forest(add_track_ids_to_tracks_df(dataframe_forest_with_time))

    expected_df = np.array(
        [[1, 0, 9, 0], [2, 10, 19, 1], [3, 10, 19, 1], [4, 5, 14, 0]]
    )

    assert np.all(df.columns == ["L", "B", "E", "P"])
    assert np.all(df.values == expected_df)


def test_tracks_df_stitching() -> None:
    """
    Input data:
                   5
         2         --
         --   4  /
     1  /    ---
    ---          \\
       \\         -- --
         --       6  7
         3
                 ---
                  8

    Output:
                 5
                 --
            2   /
         -----
     1  /      \\
    ---          -----
       \\         6
         --
         3

    """
    df = pd.DataFrame(
        [
            [1, 0, 1, 1, 1, NO_PARENT],
            [1, 1, 1, 1, 1, NO_PARENT],
            [1, 2, 1, 1, 1, NO_PARENT],
            [2, 3, 2, 2, 2, 1],
            [2, 4, 2, 2, 2, 1],
            [3, 3, 0, 0, 0, 1],
            [3, 4, 0, 0, 0, 1],
            [4, 5, 2, 2, 3, NO_PARENT],
            [4, 6, 2, 3, 3, NO_PARENT],
            [4, 7, 3, 3, 3, NO_PARENT],
            [5, 8, 3, 3, 3, 4],
            [5, 9, 4, 4, 4, 4],
            [6, 8, 2, 2, 2, 4],
            [6, 9, 1, 1, 2, 4],
            [7, 10, 1, 1, 1, NO_PARENT],
            [7, 11, 2, 1, 1, NO_PARENT],
            [8, 5, 10, 10, 10, NO_PARENT],
            [8, 6, 11, 11, 11, NO_PARENT],
            [8, 7, 12, 12, 12, NO_PARENT],
        ],
        columns=["track_id", "t", "z", "y", "x", "parent_track_id"],
    )

    graph = {1: [1, 2], 4: [5, 6]}

    stitched_df = stitch_tracks_df(graph, df, {1, 2, 3})

    assert not np.any(stitched_df["track_id"].isin({4, 7, 8}).values)
    assert not np.any(stitched_df["parent_track_id"].isin({4, 7, 8}).values)
    assert np.all(stitched_df["track_id"].isin({1, 2, 3, 5, 6}).values)
    assert np.all(stitched_df["parent_track_id"].isin({1, 2, NO_PARENT}).values)
    assert np.sum(stitched_df["track_id"] == 2) == 5
    assert np.sum(stitched_df["track_id"] == 6) == 4
