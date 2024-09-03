import logging
from types import ModuleType
from typing import Optional

import numpy as np
import pytest
import scipy.ndimage as ndi
from skimage.data import cells3d
from skimage.morphology import reconstruction

from ultrack.imgproc import Cellpose, detect_foreground, inverted_edt, robust_invert
from ultrack.imgproc.segmentation import reconstruction_by_dilation
from ultrack.utils.cuda import to_cpu

LOG = logging.getLogger(__name__)

try:
    import cupy as xp

    LOG.info("cupy found.")

except (ModuleNotFoundError, ImportError):
    import numpy as xp

    LOG.info("cupy not found using numpy.")


@pytest.mark.parametrize("np_module,channel_axis", [(xp, None), (xp, 0), (np, 2)])
def test_foreground_detection(
    np_module: ModuleType,
    channel_axis: Optional[int],
    request,
) -> None:

    cells = cells3d()
    nuclei = np_module.asarray(cells[:, 1])
    membrane = np_module.asarray(cells[:, 0])

    if channel_axis is not None:
        nuclei = np_module.stack([nuclei] * 2, axis=channel_axis)
        membrane = np_module.stack([membrane] * 2, axis=channel_axis)

    contours = robust_invert(membrane, [1, 1, 1], channel_axis=channel_axis)
    foreground = detect_foreground(
        nuclei, [1, 1, 1], sigma=50, channel_axis=channel_axis
    )

    assert contours.min() == 0.0
    assert contours.max() == 1.0

    if request.config.getoption("--show-napari-viewer"):
        import napari

        viewer = napari.Viewer()

        viewer.add_image(cells, blending="additive", channel_axis=1)
        viewer.add_labels(to_cpu(foreground))
        viewer.add_image(
            to_cpu(contours), blending="additive", colormap="gray_r", rendering="minip"
        )

        napari.run()


def test_reconstruction_by_dilation(request) -> None:
    cells = cells3d()
    membrane = cells[cells.shape[0] // 2, 0]

    seed = ndi.gaussian_filter(membrane, 5)

    iterative_bkg = reconstruction_by_dilation(seed, membrane, 250)
    exact_bkg = reconstruction(seed, membrane, method="dilation")

    if request.config.getoption("--show-napari-viewer"):
        import napari

        viewer = napari.Viewer()

        viewer.add_image(membrane, blending="additive", visible=False)
        viewer.add_image(seed, blending="additive", visible=False)
        viewer.add_image(iterative_bkg, blending="additive", colormap="red")
        viewer.add_image(exact_bkg, blending="additive", colormap="green")

        napari.run()

    np.testing.assert_array_almost_equal(iterative_bkg, exact_bkg)


def test_inverted_edt() -> None:
    mask = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], dtype=bool)
    expected_output = np.array(
        [[1.0, 1.0, 0.292893], [1.0, 0.646447, 0.209431], [0.646447, 0.292893, 0.0]]
    )
    output = inverted_edt(mask, voxel_size=(1, 2))
    np.testing.assert_array_almost_equal(output, expected_output)


def test_cellpose() -> None:
    pytest.importorskip("cellpose")
    image = np.random.rand(100, 100)
    cellpose_model = Cellpose()
    cellpose_model(image)
