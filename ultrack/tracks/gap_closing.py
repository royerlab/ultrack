from pathlib import Path
from typing import List, Optional, Tuple, Union

import pandas as pd
import zarr
from numpy.typing import ArrayLike
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from zarr.storage import Store

from ultrack.core.database import NO_PARENT
from ultrack.utils.array import create_zarr


def tracks_starts(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the starting nodes of each track in the given DataFrame.

    Parameters
    ----------
    tracks_df: pd.DataFrame
        The DataFrame containing track information.

    Returns
    -------
    pd.DataFrame
        The DataFrame containing the starting nodes of each track.

    """
    starting_tracklets = tracks_df[tracks_df["parent_track_id"] == NO_PARENT]
    starts = starting_tracklets.groupby("track_id").apply(
        lambda x: x.loc[x["t"].idxmin()]
    )
    return starts


def tracks_ends(tracks_df: pd.DataFrame) -> pd.DataFrame:
    """
    Find the ending nodes of tracks in the given DataFrame.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        The DataFrame containing the tracks information.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the ending nodes of tracks.

    """
    ending_tracks = tracks_df[~tracks_df["track_id"].isin(tracks_df["parent_track_id"])]
    ends = ending_tracks.groupby("track_id").apply(lambda x: x.loc[x["t"].idxmax()])
    return ends


def update_track_id(
    tracks_df: pd.DataFrame,
    old_track_id: int,
    new_track_id: int,
    new_track_id_parent: int,
) -> None:
    """
    Update track_id in `tracks_df` from `old_track_id` to `new_track_id`.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing track information with columns:
            "track_id" : Unique identifier for each track.
            "parent_track_id" : Identifier of the parent track in the forest.
            (Other columns may be present in the DataFrame but are not used in this function.)
    old_track_id : int
        Old track id to be replaced.
    new_track_id : int
        New track id to replace the old one.
    new_track_id_parent : int
        Existing parent of new track id.
    """
    mask = tracks_df["track_id"] == old_track_id
    tracks_df.loc[mask, "track_id"] = new_track_id
    tracks_df.loc[mask, "parent_track_id"] = new_track_id_parent
    tracks_df.loc[
        tracks_df["parent_track_id"] == old_track_id, "parent_track_id"
    ] = new_track_id


def close_tracks_gaps(
    tracks_df: pd.DataFrame,
    max_gap: int,
    max_radius: float,
    spatial_columns: List[str] = ["z", "y", "x"],
    scale: Optional[ArrayLike] = None,
    segments: Optional[ArrayLike] = None,
    segments_store_or_path: Union[Store, Path, str, None] = None,
) -> Union[pd.DataFrame, Tuple[pd.DataFrame, ArrayLike]]:
    """
    Close gaps between tracklets in the given DataFrame.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        The DataFrame containing the tracks information.
    max_gap : int
        The maximum gap size to close.
    max_radius : float
        The maximum distance between the end of one tracklet and the start of the next tracklet.
    spatial_columns : List[str]
        The names of the columns containing the spatial information.
    scale : Optional[ArrayLike]
        The scaling factors for the spatial columns.
    segments : Optional[ArrayLike]
        When provided, the function will update the segments labels to match the tracks.

    Returns
    -------
    Union[pd.DataFrame, Tuple[pd.DataFrame, ArrayLike]]
        The DataFrame containing the tracks information with the gaps closed.
    """

    if segments is None:
        out_segments = None
    else:
        raise NotImplementedError("The segments argument is not implemented yet.")
        # TODO: continue implementation
        out_segments = create_zarr(
            segments.shape,
            segments.dtype,
            segments_store_or_path,
            chunks=segments.chunks,
        )

        if isinstance(segments, zarr.Array):
            zarr.copy(segments, out_segments, if_exists="replace")
        else:
            out_segments[...] = segments

    tracks_df = tracks_df.copy()

    if scale is not None:
        tracks_df[spatial_columns] *= scale

    starts = tracks_starts(tracks_df)
    ends = tracks_ends(tracks_df)

    new_nodes = []
    track_id_map = {}

    for gap in range(1, max_gap + 1):

        starts_by_t = starts.groupby("t")

        to_remove_from_start = []
        to_remove_from_end = []

        for t, end_group in ends.groupby("t"):

            try:
                start_group = starts_by_t.get_group(t + gap + 1)
            except KeyError:
                continue

            # optimal distance matching between end and start tracklets
            dists = cdist(end_group[spatial_columns], start_group[spatial_columns])
            row_ind, col_ind = linear_sum_assignment(dists)
            valid_matches = dists[row_ind, col_ind] < max_radius

            for i, j in zip(row_ind[valid_matches], col_ind[valid_matches]):
                end_node = end_group.iloc[i]
                start_node = start_group.iloc[j]

                update_track_id(
                    tracks_df,
                    start_node["track_id"],
                    end_node["track_id"],
                    end_node["parent_track_id"],
                )

                track_id_map[start_node["track_id"].item()] = end_node[
                    "track_id"
                ].item()

                for t in range(int(end_node["t"] + 1), int(start_node["t"])):
                    new_node = end_node.copy()
                    weight = (t - end_node["t"]) / (start_node["t"] - end_node["t"])
                    step = (
                        start_node[spatial_columns] - end_node[spatial_columns]
                    ) * weight
                    new_node["t"] = t
                    new_node[spatial_columns] = end_node[spatial_columns] + step
                    new_nodes.append(new_node)

            to_remove_from_start.extend(start_group.index[col_ind[valid_matches]])
            to_remove_from_end.extend(end_group.index[row_ind[valid_matches]])

        starts = starts.drop(to_remove_from_start)
        ends = ends.drop(to_remove_from_end)

    tracks_df = pd.concat([tracks_df, pd.DataFrame(new_nodes)], ignore_index=True)
    for int_cols in ("track_id", "parent_track_id"):
        tracks_df[int_cols] = tracks_df[int_cols].astype(int)

    tracks_df = tracks_df.sort_values(["track_id", "t"])

    if scale is not None:
        tracks_df[spatial_columns] /= scale

    if out_segments is None:
        return tracks_df

    # TODO:
    #   update segments using `track_id_map` and add new segments

    return tracks_df, out_segments
