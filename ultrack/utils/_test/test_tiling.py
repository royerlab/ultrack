from typing import Tuple

import napari
import numpy as np
import pytest
import scipy.ndimage as ndi
from skimage.data import binary_blobs
from skimage.segmentation import relabel_sequential

from ultrack.utils.tiling import apply_tiled_and_stitch


def _cc_labeling(offset: int, arr: np.ndarray) -> tuple[int, np.ndarray]:
    offset = max(1, offset)
    out, _ = ndi.label(arr)
    out = relabel_sequential(out, offset=offset)[0]
    return max(out.max(), offset) + 1, out


@pytest.mark.parametrize(
    "n_dim,chunk_size",
    [
        (2, (48, 48)),
        (2, (200, 120)),
        (3, (12, 24, 24)),
    ],
)
def test_apply_tiled_and_stitch(
    n_dim: int,
    chunk_size: Tuple[int],
    request,
) -> None:
    # NOTE: Use "--show-napari-viewer" to show viewer, useful when debugging
    shape = (128, 160, 57)[:n_dim]

    in_arr = binary_blobs(
        length=max(shape),
        volume_fraction=0.1,
        n_dim=n_dim,
        blob_size_fraction=0.05,
        rng=0,
    )[tuple(slice(0, s) for s in shape)]

    out_arr = np.zeros(shape, dtype=np.int32)

    apply_tiled_and_stitch(
        in_arr,
        func=_cc_labeling,
        chunk_size=chunk_size,
        out_array=out_arr,
    )

    out_arr = relabel_sequential(out_arr)[0]
    expected_arr, expected_n_labels = ndi.label(in_arr)

    if request.config.getoption("--show-napari-viewer"):
        viewer = napari.Viewer()
        viewer.add_labels(in_arr, name="input")
        viewer.add_labels(out_arr, name="output")
        viewer.add_labels(expected_arr, name="expected")

        napari.run()

    assert out_arr.max() == expected_n_labels


if __name__ == "__main__":
    test_apply_tiled_and_stitch()
