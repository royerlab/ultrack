import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from numba import njit, typed, types
from numpy.typing import ArrayLike
from zarr.storage import Store

from ultrack.utils.constants import NO_PARENT
from ultrack.utils.segmentation import SegmentationPainter, copy_segments

LOG = logging.getLogger(__name__)


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
def _create_tracks_forest(
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

    forest = _create_tracks_forest(df.index.values, df["parent_id"].values)
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


def tracks_df_forest(
    df: pd.DataFrame,
    remove_roots: bool = False,
    numba_dict: bool = False,
) -> Union[
    Dict[int, List[int]], types.DictType(types.int64, types.ListType(types.int64))
]:
    """
    Creates the forest graph of track lineages

    Example:
    forest[parent_id] = [child_id_0, child_id_1]

    Parameters
    ----------
    df : pd.DataFrame
        Tracks dataframe.
    remove_roots : bool
        If True, removes root nodes (nodes with no parent).
    numba_dict: bool
        If True, returns a numba typed dictionary.

    Returns
    -------
    Dict[int, List[int]]
        Forest graph where parent maps to their children (parent -> children)
    """
    df = df.drop_duplicates("track_id")

    if remove_roots:
        df = df[df["parent_track_id"] != NO_PARENT]

    nb_dict = _create_tracks_forest(
        df["track_id"].to_numpy(dtype=int),
        df["parent_track_id"].to_numpy(dtype=int),
    )

    if numba_dict:
        return nb_dict

    return {k: v for k, v in nb_dict.items()}


def inv_tracks_df_forest(df: pd.DataFrame) -> Dict[int, int]:
    """
    Returns `track_id` and `parent_track_id` leaves-to-root inverted forest (set of trees) graph structure.

    Example:
    forest[child_id] = parent_id
    """
    for col in ["track_id", "parent_track_id"]:
        if col not in df.columns:
            raise ValueError(
                f"The input dataframe does not contain the column '{col}'."
            )

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
    selected_roots = set()
    for id in track_ids:

        while True:
            parent_id = inv_graph.get(id, NO_PARENT)
            if parent_id == NO_PARENT:
                break
            id = parent_id

        selected_roots.add(id)

    graph = _create_tracks_forest(
        compressed_df["track_id"].to_numpy(dtype=int),
        compressed_df["parent_track_id"].to_numpy(dtype=int),
    )
    selected_roots = np.asarray(list(selected_roots), dtype=int)

    subforest = []
    for root in selected_roots:
        subforest += left_first_search(root, graph)

    return tracks_df[tracks_df["track_id"].isin(subforest)]


def get_subtree(graph: Dict[int, int], index: int) -> Set[int]:
    """
    Returns connected component of directed graph (subtree) of `index` of tree `graph`.

    Parameters
    ----------
    graph : Dict[int, int]
        Directed graph (forest) from Parent -> Child.
    index : int
        Index of the root node.

    Returns
    -------
    Set[int]
        Set of nodes in the subtree.
    """
    component = set()
    queue = [index]
    while queue:
        index = queue.pop()
        component.add(index)
        for child in graph.get(index, []):
            queue.append(child)

    return component


def split_trees(tracks_df: pd.DataFrame) -> List[pd.DataFrame]:
    """
    Split tracks forest into trees.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing track information with columns:
            "track_id" : Unique identifier for each track.
            "parent_track_id" : Identifier of the parent track in the forest.
            (Other columns may be present in the DataFrame but are not used in this function.)

    Returns
    -------
    List[pd.DataFrame]
        List of dataframes, each representing a tree.
    """
    graph = tracks_df_forest(tracks_df, numba_dict=False)
    roots = graph.pop(NO_PARENT)

    tracks_by_id = tracks_df.groupby("track_id")

    trees = []
    for root in roots:
        subtree_ids = get_subtree(graph, root)
        # much faster than using pd.concat
        if len(subtree_ids) == 1:
            trees.append(tracks_by_id.get_group(root))
        else:
            trees.append(pd.concat([tracks_by_id.get_group(i) for i in subtree_ids]))

    return trees


def split_tracks_df_by_lineage(
    tracks_df: pd.DataFrame,
) -> List[pd.DataFrame]:
    """
    Split tracks dataframe into a list of dataframes, one for each lineage, sorted by the root track id.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        Tracks dataframe with columns:
            "track_id" : Unique identifier for each track.
            "parent_track_id" : Identifier of the parent track in the forest.
            (Other columns may be present in the DataFrame but are not used in this function.)

    Returns
    -------
    List[pd.DataFrame]
        List of dataframes, one for each lineage.
    """
    roots = tracks_df[tracks_df["parent_track_id"] == NO_PARENT]["track_id"].to_numpy()
    lineages = [get_subgraph(tracks_df, root) for root in np.sort(roots)]
    return lineages


def get_paths_to_roots(
    tracks_df: pd.DataFrame,
    graph: Optional[Dict[int, int]] = None,
    *,
    node_index: Optional[int] = None,
    track_index: Optional[int] = None,
) -> pd.DataFrame:
    """
    Returns paths from `node_index` or `track_index` to roots.
    If `node_index` and `track_index` are None, returns all paths to roots.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing track information with columns:
            "track_id" : Unique identifier for each track.
            "parent_track_id" : Identifier of the parent track in the forest.
            (Other columns may be present in the DataFrame but are not used in this function.)
    graph : Optional[Dict[int, int]], optional
        Inverted forest graph, if not provided it will be computed from `tracks_df`.
    node_index : Optional[int], optional
        Node (dataframe) index to compute path to roots.
    track_index : Optional[int], optional
        Track index (track_id column value) to compute path to roots.

    Returns
    -------
    pd.DataFrame
        DataFrame containing paths to roots.
    """
    if node_index is not None and track_index is not None:
        raise ValueError("Only one of `node_index` and `track_index` can be specified.")

    if graph is None:
        graph = inv_tracks_df_forest(tracks_df)

    if node_index is not None:
        track_indices = [tracks_df.loc[node_index, "track_id"]]

    elif track_index is not None:
        track_indices = [track_index]

    else:  # both are None, return all paths
        LOG.info(
            "Both `node_index` and `track_index` are None. Returning all paths to roots."
        )
        track_indices = tracks_df["track_id"].unique().astype(int)
        parent_ids = np.asarray(list(graph.values()), dtype=int)
        track_indices = track_indices[np.isin(track_indices, parent_ids, invert=True)]

    df_by_track_id = tracks_df.groupby("track_id")

    dfs = []
    for track_id in track_indices:

        current_dfs = []
        idx = track_id
        while idx != NO_PARENT:
            current_dfs.append(df_by_track_id.get_group(idx))
            idx = graph.get(idx, NO_PARENT)

        # reverse order so that the root is the first row
        path_df = pd.concat(reversed(current_dfs), axis=0)

        # making it a single track
        path_df["track_id"] = track_id
        if "parent_track_id" in path_df.columns:
            path_df["parent_track_id"] = NO_PARENT

        dfs.append(path_df)

    out_df = pd.concat(dfs, axis=0)

    # remove nodes after node_index
    if node_index is not None:
        index = out_df.index.get_loc(node_index)
        out_df = path_df.iloc[: index + 1]

    return out_df


def _get_children(
    track_id: int,
    graph: Dict[int, List[int]],
) -> List[int]:
    """
    Returns children of `track_id` in `graph`.

    NOTE: numba graph was not working with graph.get(track_id, []) so we use this function instead.
    """
    if track_id not in graph:
        return []
    else:
        return graph[track_id]


def _filter_short_tracks(
    parent_track_id: int,
    min_length: int,
    child2parent: Dict[int, int],
    parent2children: Dict[int, List[int]],
    track_dict: Dict[int, pd.DataFrame],
    segm_painter: Optional[SegmentationPainter] = None,
) -> None:
    """
    Recursive function to short tracks created from fake division tracks shorter than `min_length`.
    All values are updated inplace.

    Parameters
    ----------
    parent_track_id : int
        Parent track id.
    min_length : int
        Minimum track length.
    child2parent : Dict[int, int]
        Child to parent track id mapping.
    parent2children : Dict[int, List[int]]
        Parent to children track id mapping.
    track_dict : Dict[int, pd.DataFrame]
        Dictionary of track dataframes.
    segm_painter : Optional[SegmentationPainter]
        Segmentation painter to update tracks segmentation label.
    """
    children_id = _get_children(parent_track_id, parent2children)

    if len(children_id) != 0 and len(children_id) != 2:
        raise ValueError(
            f"Track {parent_track_id} has {len(children_id)} children. "
            "Only tracks with 0 or 2 children are supported."
        )

    # bottom up, first fix children
    is_short_and_childless = []
    for child_id in children_id:
        # might update children length
        _filter_short_tracks(
            child_id,
            min_length,
            child2parent,
            parent2children,
            track_dict,
            segm_painter,
        )
        # check which children could be removed
        length = len(track_dict[child_id])
        is_short_and_childless.append(
            length <= min_length and len(_get_children(child_id, parent2children)) == 0
        )

    # if there's no child or all pass the filtering criteria we do not remove them.
    if sum(is_short_and_childless) == 1:

        for child_id, to_remove in zip(children_id, is_short_and_childless):
            child_track = track_dict.pop(child_id)

            if to_remove:
                new_child_track_id = 0
            else:
                # mergin sibling track
                new_child_track_id = parent_track_id
                child_track["track_id"] = new_child_track_id
                child_track["parent_track_id"] = child2parent.get(
                    parent_track_id, NO_PARENT
                )  # parent of parent track id
                track_dict[parent_track_id] = pd.concat(
                    (track_dict[parent_track_id], child_track)
                )

                # fix sibling tracks children
                child_children_id = _get_children(child_id, parent2children)

                for child_child_id in child_children_id:
                    child_child_track = track_dict[child_child_id]
                    child_child_track["parent_track_id"] = parent_track_id
                    child2parent[child_child_id] = parent_track_id

                parent2children[parent_track_id] = child_children_id

            # relabeling
            if segm_painter is not None:
                for t in child_track["t"]:
                    segm_painter.add_relabel(t, child_id, new_child_track_id)


def filter_short_sibling_tracks(
    tracks_df: pd.DataFrame,
    min_length: int,
    segments: Optional[ArrayLike] = None,
    segments_store_or_path: Union[Store, Path, str, None] = None,
    overwrite: bool = False,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, ArrayLike]]:
    """
    Filter short tracks created from fake division tracks shorter than `min_length`.

    This function tranverse the graph bottom up and remove tracks that are shorter than `min_length`
    upon divions, merging the remaining sibling track with their parent.

    If both are shorter than `min_length`, they are not removed.

    Parameters
    ----------

    tracks_df : pd.DataFrame
        DataFrame containing track information with columns:
            "track_id" : Unique identifier for each track.
            "parent_track_id" : Identifier of the parent track in the forest.
            (Other columns may be present in the DataFrame but are not used in this function.)
    min_length : int
        Minimum track length, below this value the track is removed.
    segments : Optional[ArrayLike]
        Segmentation array to update the tracks.
    segments_store_or_path : Union[Store, Path, str, None]
        Store or path to save the new segments.
    overwrite : bool
        If True, overwrite the existing segmentation array.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, ArrayLike]]
        If `segments` is None, returns the modified tracks dataframe.
        If `segments` is provided, returns the modified tracks dataframe and the updated segments.
    """
    if segments is None:
        out_segments = None
        segm_painter = None
    else:
        out_segments = copy_segments(segments, segments_store_or_path, overwrite)
        segm_painter = SegmentationPainter(out_segments)

    parent2children = tracks_df_forest(tracks_df)
    child2parent = inv_tracks_df_forest(tracks_df)

    roots = parent2children.pop(NO_PARENT)

    track_dict = {
        track_id: group.copy()
        for track_id, group in tracks_df.groupby("track_id", as_index=False)
    }

    for root in roots:
        _filter_short_tracks(
            root,
            min_length,
            child2parent,
            parent2children,
            track_dict,
            segm_painter,
        )

    out_df = pd.concat(track_dict.values(), axis=0)

    if out_segments is None:
        return out_df

    segm_painter.apply_changes()

    return out_df, out_segments
