from typing import Dict, List, Tuple, Union

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike


def _create_forest(
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
    Accumulates the length of each root. For each branch the maximum length is considered.

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
    max_len = 0
    for child in forest[track_id]:
        child_length, child_subtree = _accumulate_length(child, track_length, forest)
        max_len = max(child_length + 1, max_len)
        subtree += child_subtree
    length = track_length[track_id] + max_len
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
        df.rename(columns={"0", "track_id"}, inplace=True)

    groups = df.groupby("track_id", as_index=False)

    length = groups.size()
    length.index = length["track_id"]
    track_ids = length["track_id"].to_numpy()
    length = length["size"].to_dict()

    forest = _create_forest(track_ids, graph)

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
