import logging
from typing import Optional

import numpy as np
import pytest
from skimage.data import cells3d

from ultrack.imgproc import Cellpose, detect_foreground, robust_invert
from ultrack.utils.cuda import to_cpu

LOG = logging.getLogger(__name__)

try:
    import cupy as xp

    LOG.info("cupy found.")

except (ModuleNotFoundError, ImportError):
    import numpy as xp

    LOG.info("cupy not found using numpy.")


@pytest.mark.parametrize("channel_axis", [None, 0, 2])
def test_foreground_detection(
    channel_axis: Optional[int],
    request,
) -> None:

    cells = cells3d()
    nuclei = xp.asarray(cells[:, 1])
    membrane = xp.asarray(cells[:, 0])

    if channel_axis is not None:
        nuclei = xp.stack([nuclei] * 2, axis=channel_axis)
        membrane = xp.stack([membrane] * 2, axis=channel_axis)

    edges = robust_invert(membrane, [1, 1, 1], channel_axis=channel_axis)
    foreground = detect_foreground(
        nuclei, [1, 1, 1], sigma=50, channel_axis=channel_axis
    )

    assert edges.min() == 0.0
    assert edges.max() == 1.0

    if request.config.getoption("--show-napari-viewer"):
        import napari

        viewer = napari.Viewer()

        viewer.add_image(cells, blending="additive", channel_axis=1)
        viewer.add_labels(to_cpu(foreground))
        viewer.add_image(
            to_cpu(edges), blending="additive", colormap="gray_r", rendering="minip"
        )

        napari.run()


def test_cellpose() -> None:
    pytest.importorskip("cellpose")
    image = np.random.rand(100, 100)
    cellpose_model = Cellpose()
    cellpose_model(image)
