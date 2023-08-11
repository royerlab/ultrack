from typing import List

import numpy as np
import pandas as pd
from numba import njit, types
from numpy.typing import ArrayLike

from ultrack.core.database import NO_PARENT
from ultrack.core.export.utils import _create_tracks_forest, inv_tracks_forest


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

    tracks_df = tracks_df.drop_duplicates("track_id")
    graph = _create_tracks_forest(
        tracks_df["track_id"].to_numpy(dtype=int),
        tracks_df["parent_track_id"].to_numpy(dtype=int),
    )
    roots = graph.pop(NO_PARENT)

    sorted_track_ids = []
    for root in roots:
        sorted_track_ids += _left_first_search(root, graph)

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

    inv_graph = inv_tracks_forest(compressed_df)
    roots = []
    for id in track_ids:

        while True:
            parent_id = inv_graph.get(id, NO_PARENT)
            if parent_id == NO_PARENT:
                break
            id = parent_id

        roots.append(id)

    graph = _create_tracks_forest(
        compressed_df["track_id"].to_numpy(dtype=int),
        compressed_df["parent_track_id"].to_numpy(dtype=int),
    )

    subforest = []
    for root in roots:
        subforest += _left_first_search(root, graph)

    return tracks_df[tracks_df["track_id"].isin(subforest)]


def tracks_profile_matrix(
    tracks_df: pd.DataFrame,
    columns: List[str],
) -> np.ndarray:
    """
    Construct a profile matrix from a pandas DataFrame containing tracks data.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing track information with columns:
            "track_id" : Unique identifier for each track.
            "t" : Time step index for each data point in the track.
            Other columns specified in 'columns' parameter, representing track attributes.
    columns : List[str]
        List of strings, specifying the columns of 'tracks_df' to use as attributes.

    Returns
    -------
    np.ndarray
        A 3D NumPy array representing the profile matrix with shape (num_attributes, num_tracks, max_timesteps),
        where 'num_attributes' is the number of attributes specified in 'columns',
        'num_tracks' is the number of unique tracks, and 'max_timesteps' is the maximum number of timesteps
        encountered among all tracks.
    """
    extra_cols = []
    for column in columns:
        if column not in tracks_df.columns:
            extra_cols.append(column)

    if len(extra_cols) > 0:
        raise ValueError(f"Columns {extra_cols} not in {tracks_df.columns}.")

    if "parent_track_id" in tracks_df.columns:
        sorted_track_ids = sort_track_ids(tracks_df)
    else:
        sorted_track_ids = tracks_df["track_id"].unique()

    num_t = int(tracks_df["t"].max(skipna=True) + 1)

    profile_matrix = np.zeros((len(sorted_track_ids), num_t, len(columns)), dtype=float)

    by_track_id = tracks_df.groupby("track_id")

    for i, track_id in enumerate(sorted_track_ids):
        group = by_track_id.get_group(track_id)
        t = group["t"].to_numpy(dtype=int)
        profile_matrix[np.full_like(t, i), t] = group[columns].values

    # move attribute axis to the first dimension
    profile_matrix = profile_matrix.transpose((2, 0, 1))
    try:
        profile_matrix = np.squeeze(profile_matrix, axis=0)
    except ValueError:
        pass

    return profile_matrix
