import numpy as np
import pytest

from ultrack.imgproc.utils import array_apply


@pytest.mark.parametrize("axis", [0, 1])
def test_array_apply_parametrized(axis):
    # Create sample data
    in_data = np.array([[1, 2, 3], [4, 5, 6]])
    out_data = np.zeros_like(in_data)

    # Define a sample function to apply
    def sample_func(arr):
        return arr + len(arr)

    array_apply(sample_func, in_data, out_data, axis=axis)
    other_axes_length = in_data.shape[1 - axis]
    assert np.array_equal(out_data, in_data + other_axes_length)
