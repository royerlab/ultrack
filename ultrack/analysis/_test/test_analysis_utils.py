from typing import Dict, List

import numpy as np
import pandas as pd
from numba import typed, types

from ultrack.analysis.utils import _left_first_search, sort_track_ids

# Import the functions to be tested
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
