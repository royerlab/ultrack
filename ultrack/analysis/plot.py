from typing import List

import numpy as np
import pandas as pd

from ultrack.analysis.utils import sort_track_ids


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
        raise ValueError(f"Columns {extra_cols} not in `tracks_df`.")

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


def plot_tracks_profile(
    tracks_df: pd.DataFrame,
    columns: List[str],
    time_units: str = "frames",
    time_scale: int = 1,
) -> np.ndarray:
    # TODO
    pass
