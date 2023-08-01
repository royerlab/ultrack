from typing import Dict, List

import numpy as np
import pandas as pd
import pytest
from numba import typed, types

from ultrack.analysis import get_subgraph, sort_track_ids
from ultrack.analysis.utils import _left_first_search
from ultrack.core.database import NO_PARENT


def _to_numba_dict(
    graph: Dict[int, List[int]],
) -> types.DictType(types.int64, types.ListType(types.int64)):
    """Convert a Python dictionary to a Numba dictionary."""

    nb_graph = typed.Dict.empty(types.int64, types.ListType(types.int64))
    for k, v in graph.items():
        children = typed.List.empty_list(types.int64)
        for child in v:
            children.append(child)
        nb_graph[k] = children

    return nb_graph


def test_left_first_search_single_node():
    # Test for a single node binary tree
    graph = _to_numba_dict({})
    result = _left_first_search(1, graph)
    assert result == [1]


def test_left_first_search_complete_binary_tree():
    # Test for a complete binary tree (all nodes have two children)
    graph = _to_numba_dict({1: [2, 3], 2: [4, 5], 3: [6, 7]})
    result = _left_first_search(1, graph)
    assert result == [4, 2, 5, 1, 6, 3, 7]


def test_sort_track_ids_forest():
    # Test for multiple binary trees forming a forest
    data = {
        "track_id": np.repeat([1, 2, 3, 4, 5, 6, 7], 3),
        "parent_track_id": np.repeat([NO_PARENT, 1, 1, NO_PARENT, 4, 4, NO_PARENT], 3),
    }

    tracks_df = pd.DataFrame(data)
    result = sort_track_ids(tracks_df)
    assert np.array_equal(result, np.array([2, 1, 3, 5, 4, 6, 7]))


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
