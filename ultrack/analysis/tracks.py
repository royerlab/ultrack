import pandas as pd


def displacement(
    tracks_df: pd.DataFrame,
    lag: int = 1,
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
    >>> print(displacement(df))
       z    y    x
    0 0.0  0.0  0.0
    1 1.0  1.0  1.0
    2 0.0  0.0  0.0
    3 2.0  1.0  0.0
    """

    tracks_df.sort_values(by=["track_id", "t"], inplace=True)

    cols = ["z", "y", "x"]
    cols = [c for c in cols if c in tracks_df.columns]

    out = tracks_df.groupby("track_id", as_index=False)[["z", "y", "x"]].diff(
        periods=lag
    )
    out.fillna(0, inplace=True)

    return out
