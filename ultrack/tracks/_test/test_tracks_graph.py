import pandas as pd
import pytest

from ultrack.core.database import NO_PARENT
from ultrack.tracks.graph import get_subgraph


@pytest.fixture
# Sample data for testing
def tracks_df() -> pd.DataFrame:
    data = {
        "track_id": [1, 2, 3, 4, 5, 6, 7, 8],
        "parent_track_id": [NO_PARENT, 1, 1, 2, 2, NO_PARENT, 6, 6],
    }
    return pd.DataFrame(data)


def test_get_subgraph_single_track_id(tracks_df: pd.DataFrame):
    track_ids = [3]
    lineage_ids = [1, 2, 3, 4, 5]
    result_df = get_subgraph(tracks_df, track_ids)
    expected_df = tracks_df[tracks_df["track_id"].isin(lineage_ids)]
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_get_subgraph_multiple_track_ids(tracks_df: pd.DataFrame):
    track_ids = [2, 3, 5]
    lineage_ids = [1, 2, 3, 4, 5]
    result_df = get_subgraph(tracks_df, track_ids)
    expected_df = tracks_df[tracks_df["track_id"].isin(lineage_ids)]
    pd.testing.assert_frame_equal(result_df, expected_df)


def test_get_subgraph_from_different_trees(tracks_df: pd.DataFrame):
    track_ids = [3, 7]
    result_df = get_subgraph(tracks_df, track_ids)
    pd.testing.assert_frame_equal(result_df, tracks_df)


def test_get_subgraph_empty_subgraph(tracks_df: pd.DataFrame):
    track_ids = [9, 10, 11]
    result_df = get_subgraph(tracks_df, track_ids)
    assert result_df.empty
