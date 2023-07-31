from typing import List

import numpy as np
import pandas as pd
from numba import njit, types

from ultrack.core.database import NO_PARENT
from ultrack.core.export.utils import _create_tracks_forest


@njit
def _left_first_search(
    track_id: int,
    graph: types.DictType(types.int64, types.ListType(types.int64)),
) -> List[int]:
    """
    Perform a left-first traversal on a binary tree represented as a graph and return a list
    of track IDs in the order they are visited during the traversal.

    Parameters
    ----------
    track_id : int
        The ID of the track to start the traversal from.
    graph : Dict[int, List[int]]
        The graph representing the binary tree. It is a dictionary where the keys are track IDs,
        and the values are lists of two child track IDs. The binary tree must have exactly two
        children for each node.

    Returns
    -------
    List[int]
        A list of track IDs visited during the left-first traversal of the binary tree,
        with `track_id` being the starting point.

    Example
    -------
    >>> graph = {1: [2, 3], 2: [4, 5], 3: [6, 7], 4: None, 5: None, 6: None, 7: None}
    >>> result = _left_first_search(1, graph)
    >>> print(result)
    [4, 2, 5, 1, 6, 3, 7]
    """

    children = graph.get(track_id)

    if children is None:
        return [track_id]

    assert len(children) == 2

    sorted_track_ids = _left_first_search(children[0], graph)
    sorted_track_ids.append(track_id)
    sorted_track_ids += _left_first_search(children[1], graph)

    return sorted_track_ids


def sort_track_ids(
    tracks_df: pd.DataFrame,
) -> np.ndarray:
    """
    Sort track IDs in a given DataFrame representing tracks in a way that maintains the left-first
    order of the binary tree formed by their parent-child relationships.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        A DataFrame containing information about tracks, where each row represents a track and
        contains at least two columns - "track_id" and "track_parent_id". The "track_id" column
        holds unique track IDs, and the "track_parent_id" column contains the parent track IDs
        for each track. The DataFrame should have a consistent parent-child relationship, forming
        one or multiple binary trees.

    Returns
    -------
    np.ndarray
        A NumPy array containing the sorted track IDs based on the left-first traversal of the
        binary trees formed by the parent-child relationships.

    Example
    -------
    >>> import pandas as pd
    >>> import numpy as np
    >>> data = {
    ...     "track_id": [1, 2, 3, 4, 5, 6, 7],
    ...     "track_parent_id": [None, 1, 1, 2, 2, 3, 3],
    ... }
    >>> tracks_df = pd.DataFrame(data)
    >>> sorted_track_ids = sort_track_ids(tracks_df)
    >>> print(sorted_track_ids)
    [4 2 5 1 6 3 7]
    """

    tracks_df = tracks_df.drop_duplicates(subset=["track_id"])
    graph = _create_tracks_forest(
        tracks_df["track_id"].values,
        tracks_df["parent_track_id"].values,
    )
    roots = graph.pop(NO_PARENT)

    sorted_track_ids = []
    for root in roots:
        sorted_track_ids += _left_first_search(root, graph)

    return sorted_track_ids
