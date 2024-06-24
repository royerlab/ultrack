from typing import Tuple

import numpy as np
import pytest
import scipy.ndimage as ndi
import zarr

from ultrack.imgproc import register_timelapse


@pytest.mark.parametrize(
    "timelapse_mock_data",
    [
        {"length": 3, "size": 32, "n_dim": 3},
    ],
    indirect=True,
)
def test_register_timelapse(
    timelapse_mock_data: Tuple[zarr.Array, zarr.Array, zarr.Array],
) -> None:
    _, moved_edges, _ = timelapse_mock_data

    shift = 8

    # adding a new to emulate a channel
    moved_edges = moved_edges[...][:, None]

    for i in range(moved_edges.shape[0]):
        moved_edges[i] = ndi.shift(moved_edges[i], (0, i * shift // 2, 0, 0), order=1)

    fixed_edges = register_timelapse(moved_edges, reference_channel=0, padding=shift)

    for i in range(moved_edges.shape[0] - 1):
        # removing padding and out of fov regions
        volume = fixed_edges[i, :, : -2 * shift]
        next_vol = fixed_edges[i + 1, :, : -2 * shift]

        np.testing.assert_allclose(volume, next_vol)
