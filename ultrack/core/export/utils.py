import logging

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)


def spatial_drift(df: pd.DataFrame, lag: int = 1) -> pd.Series:
    """Helper function to compute the drift of a dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Ordered dataframe with columns `t`, `z`, `y`, and `x`.
    lag : int, optional
        `t` lag, by default 1

    Returns
    -------
    pd.Series
        Drift values, invalid values are 0.
    """
    df = df.sort_values("t")
    drift = np.sqrt(
        np.square(df[["z", "y", "x"]] - df[["z", "y", "x"]].shift(periods=lag)).sum(
            axis=1
        )
    )
    drift.values[:lag] = 0.0
    return drift


def estimate_drift(df: pd.DataFrame, quantile: float = 0.99) -> float:
    """Compute a estimate of the tracks drift.

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
    distances = df.groupby("track_id").apply(spatial_drift)
    robust_max_distance = np.quantile(distances, quantile)
    LOG.info(f"{quantile} quantile spatial drift distance of {robust_max_distance}")
    return robust_max_distance
