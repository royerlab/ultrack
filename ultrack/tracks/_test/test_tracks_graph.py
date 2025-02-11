import numpy as np
import pandas as pd
import pytest

from ultrack.tracks import (
    filter_short_sibling_tracks,
    get_paths_to_roots,
    get_subgraph,
    split_trees,
)
from ultrack.utils.constants import NO_PARENT


@pytest.fixture
# Sample data for testing
def tracks_df() -> pd.DataFrame:
    data = {
        "track_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "parent_track_id": [NO_PARENT, 1, 1, 2, 2, NO_PARENT, 6, 6],
    }
    return pd.DataFrame(data)


def test_get_subgraph_single_track_id(tracks_df: pd.DataFrame) -> None:
    track_ids = [3]
    lineage_ids = [1, 2, 3, 4, 5]
    result_df = get_subgraph(tracks_df, track_ids)
    expected_df = tracks_df[tracks_df["track_id"].isin(lineage_ids)]
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_get_subgraph_multiple_track_ids(tracks_df: pd.DataFrame) -> None:
    track_ids = [2, 3, 5]
    lineage_ids = [1, 2, 3, 4, 5]
    result_df = get_subgraph(tracks_df, track_ids)
    expected_df = tracks_df[tracks_df["track_id"].isin(lineage_ids)]
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_get_subgraph_from_different_trees(tracks_df: pd.DataFrame) -> None:
    track_ids = [3, 7]
    result_df = get_subgraph(tracks_df, track_ids)
    pd.testing.assert_frame_equal(result_df, tracks_df)


def test_get_subgraph_empty_subgraph(tracks_df: pd.DataFrame) -> None:
    track_ids = [9, 10, 11]
    result_df = get_subgraph(tracks_df, track_ids)
    assert result_df.empty


def test_get_paths_to_roots() -> None:
    tracks_df = pd.DataFrame(
        {
            "t": [1, 2, 3, 3, 4, 4, 5, 1, 2, 2],
            "track_id": [1, 1, 2, 3, 4, 5, 5, 6, 7, 8],
            "parent_track_id": [NO_PARENT, NO_PARENT, 1, 1, 2, 2, 2, NO_PARENT, 6, 6],
        },
        index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    )
    path_df = get_paths_to_roots(tracks_df, track_index=5)

    single_path_df = pd.DataFrame(
        {
            "t": [1, 2, 3, 4, 5],
            "track_id": [5] * 5,
            "parent_track_id": [NO_PARENT] * 5,
        },
        index=[1, 2, 3, 6, 7],
    )

    pd.testing.assert_frame_equal(
        path_df,
        single_path_df,
    )

    path_df = get_paths_to_roots(tracks_df, node_index=6)

    single_path_df = pd.DataFrame(
        {
            "t": [1, 2, 3, 4],
            "track_id": [5] * 4,
            "parent_track_id": [NO_PARENT] * 4,
        },
        index=[1, 2, 3, 6],
    )

    pd.testing.assert_frame_equal(
        path_df,
        single_path_df,
    )

    path_df = get_paths_to_roots(tracks_df)
    all_paths_df = pd.DataFrame(
        {
            "t": [1, 2, 3, 1, 2, 3, 4, 1, 2, 3, 4, 5, 1, 2, 1, 2],
            "track_id": [3, 3, 3, 4, 4, 4, 4, 5, 5, 5, 5, 5, 7, 7, 8, 8],
        },
        index=[1, 2, 4, 1, 2, 3, 5, 1, 2, 3, 6, 7, 8, 9, 8, 10],
    )
    all_paths_df["parent_track_id"] = NO_PARENT

    pd.testing.assert_frame_equal(
        path_df,
        all_paths_df,
    )


def test_filter_short_sibling_tracks() -> None:
    """
    Input:
              4 -
             /
        / 3 -      6 ----
        |    \\    /
        |     5 -       8 -
        |         \\    /
    1 --|          7 -
        |              \
        |               9 -
        |
        \\ 2 -


    Output:
            6 ----
           /
    1 ----       8 -
           \\    /
            7 -
                \
                 9 -
    """
    segm = np.zeros((8, 6), dtype=np.uint8)

    tracks_df = pd.DataFrame(
        {
            "t": [0, 1, 2, 2, 3, 3, 4, 5, 6, 7, 4, 5, 5],
            "x": [2, 2, 0, 4, 5, 3, 4, 4, 4, 4, 2, 3, 1],
            "track_id": [1, 1, 2, 3, 4, 5, 6, 6, 6, 6, 7, 8, 9],
            "parent_track_id": [NO_PARENT, NO_PARENT, 1, 1, 3, 3, 5, 5, 5, 5, 5, 7, 7],
        }
    )
    segm[tracks_df["t"].astype(int), tracks_df["x"].astype(int)] = tracks_df[
        "track_id"
    ].values

    filtered_df, filtered_segm = filter_short_sibling_tracks(
        tracks_df, min_length=3, segments=segm
    )
    expected_df = pd.DataFrame(
        {
            "t": [0, 1, 2, 3, 4, 5, 6, 7, 4, 5, 5],
            "x": [2, 2, 4, 3, 4, 4, 4, 4, 2, 3, 1],
            "track_id": [1, 1, 1, 1, 6, 6, 6, 6, 7, 8, 9],
            "parent_track_id": [
                NO_PARENT,
                NO_PARENT,
                NO_PARENT,
                NO_PARENT,
                1,
                1,
                1,
                1,
                1,
                7,
                7,
            ],
        },
        index=[0, 1, 3, 5, 6, 7, 8, 9, 10, 11, 12],
    )
    expected_segm = np.zeros((8, 6), dtype=np.uint8)
    expected_segm[
        expected_df["t"].astype(int), expected_df["x"].astype(int)
    ] = expected_df["track_id"].values

    pd.testing.assert_frame_equal(
        filtered_df,
        expected_df,
    )
    np.testing.assert_array_equal(
        filtered_segm,
        expected_segm,
    )


def test_split_trees() -> None:
    tracks_df = pd.DataFrame(
        {
            "track_id": [1, 2, 3, 4, 4],
            "parent_track_id": [NO_PARENT, 1, 1, NO_PARENT, NO_PARENT],
        }
    )

    forest = split_trees(tracks_df)
    expected_forest = [
        pd.DataFrame(
            {
                "track_id": [1, 2, 3],
                "parent_track_id": [NO_PARENT, 1, 1],
            },
            index=[0, 1, 2],
        ),
        pd.DataFrame(
            {
                "track_id": [4, 4],
                "parent_track_id": [NO_PARENT, NO_PARENT],
            },
            index=[3, 4],
        ),
    ]

    for tree, expected_tree in zip(forest, expected_forest):
        pd.testing.assert_frame_equal(
            tree,
            expected_tree,
        )
