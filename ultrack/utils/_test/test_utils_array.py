from typing import Tuple

import numpy as np
import pytest

from ultrack.config import MainConfig
from ultrack.utils.array import array_apply
from ultrack.utils.ultrack_array import UltrackArray


@pytest.mark.parametrize("axis", [0, 1])
def test_array_apply_parametrized(axis):
    # Create sample data
    in_data = np.array([[1, 2, 3], [4, 5, 6]])
    out_data = np.zeros_like(in_data)

    # Define a sample function to apply
    def sample_func(arr_1, arr_2):
        return arr_1 + arr_2 + len(arr_1)

    array_apply(in_data, in_data, out_array=out_data, func=sample_func, axis=axis)
    other_axes_length = in_data.shape[1 - axis]
    assert np.array_equal(out_data, 2 * in_data + other_axes_length)


@pytest.mark.parametrize(
    "key,timelapse_mock_data",
    [
        (1, {"n_dim": 3}),
        (1, {"n_dim": 2}),
        ((slice(None), 1), {"n_dim": 3}),
        ((slice(None), 1), {"n_dim": 2}),
        ((0, [1, 2]), {"n_dim": 3}),
        ((0, [1, 2]), {"n_dim": 2}),
        # ((-1, np.asarray([0, 3])),{'n_dim':3}),       #does testing negative time make sense?
        # ((-1, np.asarray([0, 3])),{'n_dim':2}),
        ((slice(1), -2), {"n_dim": 3}),
        ((slice(1), -2), {"n_dim": 2}),
        ((np.asarray(0),), {"n_dim": 3}),
        ((np.asarray(0),), {"n_dim": 2}),
        ((0, 0, slice(32)), {"n_dim": 3}),
        ((0, 0, slice(32)), {"n_dim": 2}),
    ],
    indirect=[
        "timelapse_mock_data",
    ],
)
def test_ultrack_array(
    segmentation_database_mock_data: MainConfig,
    key: Tuple,
):
    ua = UltrackArray(segmentation_database_mock_data)
    ua_numpy = ua[slice(None)]
    np.testing.assert_equal(ua_numpy[key], ua[key])
