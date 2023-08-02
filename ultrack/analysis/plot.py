from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter

from ultrack.analysis.utils import get_subgraph, sort_track_ids


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


def plot_tracks_profile(
    tracks_df: pd.DataFrame,
    columns: List[str],
    time_units: str = "frames",
    time_scale: int = 1,
) -> plt.Figure:
    """
    Plot the profiles of track attributes over time.

    Parameters
    ----------
    tracks_df : pd.DataFrame
        DataFrame containing track information with columns:
            "track_id" : Unique identifier for each track.
            "t" : Time step index for each data point in the track.
            Other columns specified in 'columns' parameter, representing track attributes.
    columns : List[str]
        List of strings, specifying the columns of 'tracks_df' to plot as track attributes.
    time_units : str, optional
        The units of time represented by the "t" column in 'tracks_df'. Default is "frames".
    time_scale : int, optional
        The scaling factor for the time axis. Default is 1, which means no scaling.

    Returns
    -------
    plt.Figure
        A Matplotlib Figure containing subplots of the profiles of track attributes over time.

    Examples
    --------
    >>> plot_tracks_profile(tracks_df, ["velocity", "acceleration"], time_units="seconds", time_scale=30)

    Notes
    -----
    This function generates a plot showing the profiles of specified track attributes over time.
    The x-axis represents time (timesteps) with labels scaled by the 'time_scale' factor.
    The y-axis represents individual track IDs, and each subplot represents one attribute.
    The function internally uses 'tracks_profile_matrix' to construct the attribute profiles.

    Parameters 'time_units' and 'time_scale' are used to control the display of the time axis.
    The time axis can be displayed in various units (e.g., frames, seconds) and scaled accordingly.

    The function returns a Matplotlib Figure, which can be further customized if needed.
    """

    # Use 'g' format to remove trailing zeros if any
    formatter = FuncFormatter(lambda x, _: f"{x * time_scale:g}")
    prev_formatter = plt.gca().xaxis.get_major_formatter()
    plt.gca().xaxis.set_major_formatter(formatter)

    profiles = tracks_profile_matrix(tracks_df, columns)

    fig, axs = plt.subplots(len(columns), 1, figsize=(100, 100 * len(columns)))

    vmax = profiles.max()

    for i, column in enumerate(columns):
        axs[i].imshow(
            profiles[i],
            cmap="magma",
            interpolation="nearest",
            aspect="auto",
            vmin=0,
            vmax=vmax,
        )
        axs[i].set_title(column)
        axs[i].set_xlabel(f"Time ({time_units})")
        axs[i].set_ylabel("Track ID")
        axs[i].get_yaxis().set_visible(False)

    plt.gca().xaxis.set_major_formatter(prev_formatter)

    return fig


if __name__ == "__main__":
    import sys

    tracks_df = pd.read_csv(sys.argv[1])
    tracks_df = get_subgraph(tracks_df, [6207, 6533])
    columns = [
        c
        for c in tracks_df.columns
        if c not in ["track_id", "t", "parent_track_id", "id", "parent_id"]
    ]
    columns = [
        "intensity_mean_0",
        "intensity_mean_1",
        "intensity_mean_2",
    ]
    fig = plot_tracks_profile(tracks_df, columns)
    plt.show()
