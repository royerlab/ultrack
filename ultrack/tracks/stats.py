import logging
from typing import List, Optional

import numpy as np
import pandas as pd

from ultrack.tracks.sorting import sort_track_ids
from ultrack.utils.constants import NO_PARENT

LOG = logging.getLogger(__name__)


def tracks_df_movement(
    tracks_df: pd.DataFrame,
    lag: int = 1,
    cols: Optional[tuple[str, ...]] = None,
) -> pd.DataFrame:
    """
    Compute the displacement for track data across given time lags.

    This function computes the displacement (difference) for track coordinates
    across the specified lag periods.

    NOTE: this sort the dataframe by ["track_id", "t"].

    Parameters
    ----------
    tracks_df : pd.DataFrame
        Dataframe containing track data. It is expected to have columns
        ["track_id", "t"] and any of ["z", "y", "x"] representing the 3D coordinates.

    lag : int, optional
        Number of periods to compute the difference over. Default is 1.

    cols : tuple[str, ...], optional
        Columns to compute the displacement for. If not provided, it will try to
        find any of ["z", "y", "x"] columns in the dataframe and use them.

    Returns
    -------
    pd.DataFrame
        Dataframe of the displacement (difference) of coordinates for the given lag.
        Displacements for the first row of each track_id will be set to zero.

    Examples
    --------
    >>> df = pd.DataFrame({
    ...     "track_id": [1, 1, 2, 2],
    ...     "t": [1, 2, 1, 2],
    ...     "z": [0, 1, 0, 2],
    ...     "y": [1, 2, 1, 2],
    ...     "x": [2, 3, 2, 2]
    ... })
    >>> print(tracks_df_movement(df))
       z    y    x
    0 0.0  0.0  0.0
    1 1.0  1.0  1.0
    2 0.0  0.0  0.0
    3 2.0  1.0  0.0
    """
    if "track_id" not in tracks_df.columns:
        tracks_df["track_id"] = 0

    tracks_df.sort_values(by=["track_id", "t"], inplace=True)

    if cols is None:
        cols = []
        for c in ["z", "y", "x"]:
            if c in tracks_df.columns:
                cols.append(c)
    else:
        cols = list(cols)

    out = tracks_df.groupby("track_id", as_index=False)[cols].diff(periods=lag)
    out.fillna(0, inplace=True)

    return out


def estimate_drift(df: pd.DataFrame, quantile: float = 0.99) -> float:
    """Compute a estimate of the tracks drift.

    NOTE: this sort the dataframe by ["track_id", "t"].

    Parameters
    ----------
    df : pd.DataFrame
        Tracks dataframe, must have `track_id` column.
    quantile : float, optional
        Drift quantile, by default 0.99

    Returns
    -------
    float
        Drift from the given quantile.
    """
    displacement = tracks_df_movement(df)
    assert all(c in ["z", "y", "x"] for c in displacement.columns)
    distances = np.linalg.norm(displacement, axis=1)
    robust_max_distance = np.quantile(distances, quantile)
    LOG.info(f"{quantile} quantile spatial drift distance of {robust_max_distance}")
    return robust_max_distance


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


def tracks_length(
    tracks_df: pd.DataFrame,
    include_appearing: bool = True,
    include_disappearing: bool = True,
) -> pd.DataFrame:
    """
    Compute the length of each track in a tracks dataframe.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing track information with columns:
            "t" : Time step index for each data point in the track.
            "track_id" : Unique identifier for each track.
            "parent_track_id" : Unique identifier for the parent track.
    include_appearing : bool, optional
        Include tracks that appear outside the first time step, by default True.
    include_disappearing : bool, optional
        Include tracks that disappear outside the last time step, by default True.

    Returns
    -------
    pd.DataFrame
        Series containing the length of each track.
    """
    for c in ("t", "track_id", "parent_track_id"):
        if c not in tracks_df.columns:
            raise ValueError(f"Column '{c}' not in {tracks_df.columns}.")

    tracks_df = tracks_df.groupby("track_id", as_index=False).agg(
        start=pd.NamedAgg("t", lambda x: x.min()),
        end=pd.NamedAgg("t", lambda x: x.max()),
        parent_track_id=pd.NamedAgg("parent_track_id", lambda x: x.iloc[0]),
    )
    tracks_df["length"] = tracks_df["end"] - tracks_df["start"] + 1

    t_max = tracks_df["end"].max()

    if not include_appearing:
        tracks_df = tracks_df[
            (tracks_df["start"] == 0) | (tracks_df["parent_track_id"] != NO_PARENT)
        ]

    if not include_disappearing:
        tracks_df = tracks_df[
            (tracks_df["end"] == t_max)
            | (tracks_df["track_id"].isin(tracks_df["parent_track_id"]))
        ]

    return tracks_df
