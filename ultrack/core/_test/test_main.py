from typing import Tuple

import pytest
import zarr

from ultrack import track
from ultrack.config import MainConfig


@pytest.mark.parametrize(
    "config_content,timelapse_mock_data",
    [
        ({"segmentation.n_workers": 4}, {"length": 4, "size": 128, "n_dim": 2}),
        ({"segmentation.n_workers": 4}, {"length": 4, "size": 64, "n_dim": 3}),
    ],
    indirect=True,
)
def test_tracking(
    config_instance: MainConfig,
    timelapse_mock_data: Tuple[zarr.Array, zarr.Array, zarr.Array],
) -> None:
    foreground, contours, labels = timelapse_mock_data

    track(config_instance, labels=labels)
    track(config_instance, foreground=foreground, contours=contours, overwrite=True)
