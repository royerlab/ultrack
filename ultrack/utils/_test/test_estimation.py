from typing import Sequence, Tuple

import numpy as np
import pandas as pd
import pytest
import zarr

from ultrack.utils.estimation import (
    estimate_drift,
    estimate_parameters_from_labels,
    spatial_drift,
)


def spatial_df(group_drift: Sequence[int], length_per_group: int = 10) -> pd.DataFrame:
    rng = np.random.default_rng(42)

    df = []
    for d in group_drift:
        pos = np.empty((length_per_group, 4))
        pos[:, 0] = np.arange(length_per_group)
        pos[0, 1:] = rng.uniform(size=3)

        for i in range(length_per_group - 1):
            step = rng.uniform(size=3)
            pos[i + 1, 1:] = pos[i, 1:] + d * step / np.linalg.norm(step)
        df.append(pd.DataFrame(pos, columns=["t", "z", "y", "x"]))
    df = pd.concat(df)
    return df


@pytest.mark.parametrize(
    "drift",
    [0.5, 5, 10],
)
def test_spatial_drift(drift: float) -> None:
    df = spatial_df([drift])
    drift_df = spatial_drift(df, lag=1)
    assert np.allclose(drift_df.values[1:], drift)
    assert np.allclose(drift_df.values[:1], 0)


@pytest.mark.parametrize(
    "length_per_group",
    [3, 8, 15],
)
def test_maximum_distance_estimate(length_per_group: int) -> None:
    group_drift = [1, 10, 20]
    df = spatial_df(group_drift, length_per_group)
    df["track_id"] = np.repeat(np.arange(len(group_drift)), length_per_group)
    assert np.allclose(estimate_drift(df), max(group_drift))


def test_timelapse_parameter_estimation(
    timelapse_mock_data: Tuple[zarr.Array, zarr.Array, zarr.Array],
) -> None:
    _, _, labels = timelapse_mock_data
    df = estimate_parameters_from_labels(labels, is_timelapse=True)
    assert not np.any(df["area"] == np.nan)
    # there's zero drift between time points on mock data
    assert np.all(df["distance"] == 0.0)


@pytest.mark.parametrize(
    "segmentation_mock_data",
    [{"n_dim": 2}, {"n_dim": 3}],
    indirect=True,
)
def test_stack_parameter_estimation(
    segmentation_mock_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    _, _, labels = segmentation_mock_data
    df = estimate_parameters_from_labels(labels, is_timelapse=False)
    assert not np.any(df["area"] == np.nan)
