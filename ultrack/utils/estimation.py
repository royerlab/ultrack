import logging

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from skimage.measure import regionprops_table
from tqdm import tqdm

LOG = logging.getLogger(__name__)

_RENAME_COLUMNS = {
    "centroid-0": "z",
    "centroid-1": "y",
    "centroid-2": "x",
    "label": "track_id",
}


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
    delta = df[["z", "y", "x"]] - df[["z", "y", "x"]].shift(periods=lag)
    drift = np.linalg.norm(delta, axis=1)
    drift[:lag] = 0.0
    return pd.Series(drift)


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


def _to_dataframe(labels: ArrayLike) -> pd.DataFrame:
    labels = np.asarray(labels)
    if labels.ndim == 2:
        labels = np.expand_dims(labels, 0)
    return pd.DataFrame(
        regionprops_table(labels, properties=["label", "centroid", "area"])
    )


def estimate_parameters_from_labels(
    labels: ArrayLike,
    is_timelapse: bool,
) -> pd.DataFrame:
    """
    Estimates `area` (per segment) and `distance`
    (between subsequent centroids of the same label) parameters from labels.

    Parameters
    ----------
    labels : ArrayLike
        Array of labels, segments belonging to the same track must have the same label.
    is_time_lapse : bool
        Indicates if it's a timelapse or a single stack.

    Returns
    -------
    pd.DataFrame
        Dataframe with `area` and `distance` colums.
    """
    if not is_timelapse:
        df = _to_dataframe(labels)
    else:
        df = []
        for t in tqdm(range(labels.shape[0]), "Estimating params."):
            _df = _to_dataframe(labels[t])
            _df["t"] = t
            df.append(_df)

        df = pd.concat(df)

    df.columns = [_RENAME_COLUMNS.get(c, c) for c in df.columns]

    if "t" in df.columns:
        distance = df.groupby("track_id", sort=True).apply(spatial_drift)
        df.sort_values(["track_id", "t"], inplace=True)
        df["distance"] = distance.values

    return df
