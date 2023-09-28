from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd
from numba import njit, typed, types
from numpy.typing import ArrayLike

from ultrack.core.database import NO_PARENT


@njit
def _fast_path_transverse(
    node: int,
    track_id: int,
    queue: List[Tuple[int, int]],
    forest: Dict[int, Tuple[int]],
) -> List[int]:
    """Transverse a path in the forest directed graph and add path (track) split into queue.

    Parameters
    ----------
    node : int
        Source path node.
    track_id : int
        Reference track id for path split.
    queue : List[Tuple[int, int]]
        Source nodes and path (track) id reference queue.
    forest : Dict[int, Tuple[int]]
        Directed graph (tree) of paths relationships.

    Returns
    -------
    List[int]
        Sequence of nodes in the path.
    """
    path = typed.List.empty_list(types.int64)

    while True:
        path.append(node)

        children = forest.get(node)
        if children is None:
            # end of track
            break

        elif len(children) == 1:
            node = children[0]

        elif len(children) == 2:
            queue.append((children[1], track_id))
            queue.append((children[0], track_id))
            break

        else:
            raise RuntimeError(
                "Something is wrong. Found node with more than two children when parsing tracks."
            )

    return path


@njit
def _fast_forest_transverse(
    roots: List[int],
    forest: Dict[int, List[int]],
) -> Tuple[List[List[int]], List[int], List[int], List[int]]:
    """Transverse the tracks forest graph creating a distinc id to each path.

    Parameters
    ----------
    roots : List[int]
        Forest roots.
    forest : Dict[int, List[int]]
        Graph (forest).

    Returns
    -------
    Tuple[List[List[int]], List[int], List[int], List[int]]
        Sequence of paths, their respective track_id, parent_track_id and length.
    """
    track_id = 1
    paths = []
    track_ids = []  # equivalent to arange
    parent_track_ids = []
    lengths = []

    for root in roots:
        queue = [(root, NO_PARENT)]

        while queue:
            node, parent_track_id = queue.pop()
            path = _fast_path_transverse(node, track_id, queue, forest)
            paths.append(path)
            track_ids.append(track_id)
            parent_track_ids.append(parent_track_id)
            lengths.append(len(path))
            track_id += 1

    return paths, track_ids, parent_track_ids, lengths


@njit
def create_tracks_forest(
    node_ids: np.ndarray, parent_ids: np.ndarray
) -> Dict[int, List[int]]:
    """Creates the forest graph of track lineages

    Parameters
    ----------
    node_ids : np.ndarray
        Nodes indices.
    parent_ids : np.ndarray
        Parent indices.

    Returns
    -------
    Dict[int, List[int]]
        Forest graph where parent maps to their children (parent -> children)
    """
    forest = {}
    for parent in parent_ids:
        forest[parent] = typed.List.empty_list(types.int64)

    for i in range(len(parent_ids)):
        forest[parent_ids[i]].append(node_ids[i])

    return forest


def add_track_ids_to_tracks_df(df: pd.DataFrame) -> pd.DataFrame:
    """Adds `track_id` and `parent_track_id` columns to forest `df`.
    Each maximal path receveis a unique `track_id`.

    Parameters
    ----------
    df : pd.DataFrame
        Forest defined by the `parent_id` column and the dataframe indices.

    Returns
    -------
    pd.DataFrame
        Inplace modified input dataframe with additional columns.
    """
    assert df.shape[0] > 0

    df.index = df.index.astype(int)
    df["parent_id"] = df["parent_id"].astype(int)

    forest = create_tracks_forest(df.index.values, df["parent_id"].values)
    roots = forest.pop(NO_PARENT)

    df["track_id"] = NO_PARENT
    df["parent_track_id"] = NO_PARENT

    paths, track_ids, parent_track_ids, lengths = _fast_forest_transverse(roots, forest)

    paths = np.concatenate(paths)
    df.loc[paths, "track_id"] = np.repeat(track_ids, lengths)
    df.loc[paths, "parent_track_id"] = np.repeat(parent_track_ids, lengths)

    unlabeled_tracks = df["track_id"] == NO_PARENT
    assert not np.any(
        unlabeled_tracks
    ), f"Something went wrong. Found unlabeled tracks\n{df[unlabeled_tracks]}"

    return df


def tracks_df_forest(df: pd.DataFrame) -> Dict[int, List[int]]:
    """
    Returns `track_id` and `parent_track_id` root-to-leaves forest (set of trees) graph structure.

    Example:
    forest[parent_id] = [child_id_0, child_id_1]
    """
    df = df.drop_duplicates("track_id")
    df = df[df["parent_track_id"] != NO_PARENT]
    graph = {}
    for parent_id, id in zip(df["parent_track_id"], df["track_id"]):
        graph[parent_id] = graph.get(parent_id, []) + [id]
    return graph


def inv_tracks_df_forest(df: pd.DataFrame) -> Dict[int, int]:
    """
    Returns `track_id` and `parent_track_id` leaves-to-root inverted forest (set of trees) graph structure.

    Example:
    forest[child_id] = parent_id
    """
    df = df.drop_duplicates("track_id")
    df = df[df["parent_track_id"] != NO_PARENT]
    graph = {}
    for parent_id, id in zip(df["parent_track_id"], df["track_id"]):
        graph[id] = parent_id
    return graph


@njit
def left_first_search(
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

    sorted_track_ids = left_first_search(children[0], graph)
    sorted_track_ids.append(track_id)
    sorted_track_ids += left_first_search(children[1], graph)

    return sorted_track_ids


def get_subgraph(
    tracks_df: pd.DataFrame,
    track_ids: ArrayLike,
) -> pd.DataFrame:
    """
    Get a subgraph from a forest of tracks represented as a DataFrame.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing track information with columns:
            "track_id" : Unique identifier for each track.
            "parent_track_id" : Identifier of the parent track in the forest.
            (Other columns may be present in the DataFrame but are not used in this function.)
    track_ids : ArrayLike
        An array-like object containing the track IDs for which to extract the subgraph.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the subgraph of tracks corresponding to the input track IDs.

    Examples
    --------
    >>> subgraph_df = get_subgraph(tracks_df, [3, 7, 10])

    Notes
    -----
    The input DataFrame 'tracks_df' should have at least two columns: "track_id" and "parent_track_id",
    where "track_id" represents the unique identifier for each track, and "parent_track_id" represents
    the identifier of the parent track in the forest.
    """
    track_ids = np.atleast_1d(track_ids).astype(int)
    compressed_df = tracks_df.drop_duplicates("track_id")

    inv_graph = inv_tracks_df_forest(compressed_df)
    roots = []
    for id in track_ids:

        while True:
            parent_id = inv_graph.get(id, NO_PARENT)
            if parent_id == NO_PARENT:
                break
            id = parent_id

        roots.append(id)

    graph = create_tracks_forest(
        compressed_df["track_id"].to_numpy(dtype=int),
        compressed_df["parent_track_id"].to_numpy(dtype=int),
    )
    roots = np.asarray(roots, dtype=int)

    subforest = []
    for root in roots:
        subforest += left_first_search(root, graph)

    return tracks_df[tracks_df["track_id"].isin(subforest)]


def get_subtree(graph: Dict[int, int], index: int) -> Set[int]:
    """Returns connected component of directed graph (subtree) of `index` of tree `graph`."""
    component = set()
    queue = [index]
    while queue:
        index = queue.pop()
        component.add(index)
        for child in graph.get(index, []):
            queue.append(child)

    return component
