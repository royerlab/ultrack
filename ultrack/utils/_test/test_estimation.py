from typing import Tuple

import numpy as np
import pytest
import zarr

from ultrack.utils.estimation import estimate_parameters_from_labels


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
