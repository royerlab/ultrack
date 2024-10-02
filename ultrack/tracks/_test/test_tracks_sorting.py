from typing import Dict, List

import numpy as np
import pandas as pd
from numba import typed, types

from ultrack.tracks import (
    left_first_search,
    sort_track_ids,
    sort_trees_by_length,
    sort_trees_by_max_radius,
)
from ultrack.utils.constants import NO_PARENT


def test_sortrees_by_length() -> None:
    """
                 5
                 --
            2   /
         -----
     1  /      \\
    ---          -----
       \\         6
         --
         3

        --------
        4
    """
    df = pd.DataFrame(
        {
            "track_id": [
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                5,
                5,
                6,
                6,
                6,
                6,
                6,
            ],
        }
    )
    graph = {6: 2, 5: 2, 2: 1, 3: 1}

    sorted = sort_trees_by_length(df, graph)

    assert len(sorted) == 2

    for track_id in [1, 2, 3, 5, 6]:
        assert track_id in sorted[0]["track_id"]
        assert track_id not in sorted[1]["track_id"]

    assert not np.any(4 == sorted[0]["track_id"])
    assert np.all(4 == sorted[1]["track_id"])

    reconstr_df = pd.concat(sorted).sort_values("track_id")
    assert np.allclose(reconstr_df, df)


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


def test_left_first_search_single_node() -> None:
    # Test for a single node binary tree
    graph = _to_numba_dict({})
    result = left_first_search(1, graph)
    assert result == [1]


def test_left_first_search_complete_binary_tree() -> None:
    # Test for a complete binary tree (all nodes have two children)
    graph = _to_numba_dict({1: [2, 3], 2: [4, 5], 3: [6, 7]})
    result = left_first_search(1, graph)
    assert result == [4, 2, 5, 1, 6, 3, 7]


def test_sort_track_ids_forest() -> None:
    # Test for multiple binary trees forming a forest
    data = {
        "track_id": np.repeat([1, 2, 3, 4, 5, 6, 7], 3),
        "parent_track_id": np.repeat([NO_PARENT, 1, 1, NO_PARENT, 4, 4, NO_PARENT], 3),
    }

    tracks_df = pd.DataFrame(data)
    result = sort_track_ids(tracks_df)
    assert np.array_equal(result, np.array([2, 1, 3, 5, 4, 6, 7]))


def test_sort_trees_by_max_radius() -> None:
    tracks_df = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 3, 4, 5, 6, 7],
            "parent_track_id": [NO_PARENT, NO_PARENT, NO_PARENT, 2, 2, NO_PARENT, 5, 5],
            "t": [0, 1, 0, 1, 1, 0, 1, 1],
            "x": [1, 2, 0, 1, 3, 7, 1, 2],
        }
    )

    expected_track_ids = [[2, 3, 4], [5, 6, 7], [1, 1]]

    sorted_tracks = sort_trees_by_max_radius(tracks_df, scale=1.0)

    for track_ids, expected_ids in zip(sorted_tracks, expected_track_ids):
        assert np.array_equal(track_ids["track_id"].to_numpy(), expected_ids)

    # no radius
    sorted_tracks = sort_trees_by_max_radius(tracks_df)

    for track_ids, expected_ids in zip(sorted_tracks, expected_track_ids):
        assert np.array_equal(track_ids["track_id"].to_numpy(), expected_ids)
