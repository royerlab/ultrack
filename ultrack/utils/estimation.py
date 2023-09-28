import logging

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from skimage.measure import regionprops_table
from tqdm import tqdm

from ultrack.tracks.stats import tracks_df_movement

LOG = logging.getLogger(__name__)

_RENAME_COLUMNS = {
    "centroid-0": "z",
    "centroid-1": "y",
    "centroid-2": "x",
    "label": "track_id",
}


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
        df["distance"] = np.linalg.norm(tracks_df_movement(df, lag=1), axis=1)

    return df
