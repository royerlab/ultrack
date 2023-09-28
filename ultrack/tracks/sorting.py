from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from ultrack.core.database import NO_PARENT
from ultrack.tracks.graph import create_tracks_forest, left_first_search


def _invert_forest(
    track_ids: np.ndarray,
    digraph: Dict[int, int],
) -> Dict[int, List[int]]:
    """
    Converts the directed graph (child -> parent) to a rooted forest (parent -> children).

    Parameters
    ----------
    track_ids : np.ndarray
        Array of track ids.
    digraph : Dict[int, int]
        Child -> parent directed graph (forest).

    Returns
    -------
    Dict[int, List[int]]
        Rooted forest (parent -> children)
    """
    forest = {}

    for id in track_ids:
        forest[id] = []

    for child, parent in digraph.items():
        forest[parent].append(child)

    return forest


def _accumulate_length(
    track_id: int,
    track_length: Dict[int, int],
    forest: Dict[int, List[int]],
) -> Tuple[int, List[int]]:
    """
    Accumulates the length of each root. For each branch the average length is considered.

    Parameters
    ----------
    track_id : int
        Current track id.
    track_length : Dict[int, int]
        Dictionary of tracklet lengths.
    forest : Dict[int, List[int]]
        Rooted forest.

    Returns
    -------
    Tuple[int, List[int]]
        Current subtree length and subtree ids.
    """
    subtree = [track_id]
    length = track_length[track_id]
    children = forest[track_id]

    if not children:
        return length, subtree

    branch_length = 0
    for child in children:
        child_length, child_subtree = _accumulate_length(child, track_length, forest)
        branch_length += child_length
        subtree += child_subtree

    length += branch_length / len(children) + 1

    return length, subtree


def sort_trees_by_length(
    df: Union[ArrayLike, pd.DataFrame],
    graph: Dict[int, int],
) -> List[pd.DataFrame]:
    """Sorts trees from the track graph by length (deepest tree path).

    Parameters
    ----------
    df : pd.DataFrame
        tracks dataframe.
    graph : Dict[int, int]
        Child -> parent tracks graph.

    Returns
    -------
    List[pd.DataFrame]
        Sorted list of tracks dataframe.
    """
    if not isinstance(df, pd.DataFrame):
        df = pd.DataFrame(df)
        df.rename(columns={0: "track_id"}, inplace=True)

    groups = df.groupby("track_id", as_index=False)

    length = groups.size()
    length.index = length["track_id"]
    track_ids = length["track_id"].to_numpy()
    length = length["size"].to_dict()

    forest = _invert_forest(track_ids, graph)

    roots = track_ids[np.isin(track_ids, list(graph.keys()), invert=True)]
    lengths = []
    subtrees = []

    for root in roots:
        tree_len, ids = _accumulate_length(root, length, forest)
        lengths.append(tree_len)
        subtrees.append(ids)

    sorted_trees = sorted(zip(lengths, subtrees), reverse=True)
    return [
        pd.concat([groups.get_group(i) for i in subtree]) for _, subtree in sorted_trees
    ]


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

    tracks_df = tracks_df.drop_duplicates("track_id")
    graph = create_tracks_forest(
        tracks_df["track_id"].to_numpy(dtype=int),
        tracks_df["parent_track_id"].to_numpy(dtype=int),
    )
    roots = graph.pop(NO_PARENT)

    sorted_track_ids = []
    for root in roots:
        sorted_track_ids += left_first_search(root, graph)

    return sorted_track_ids
