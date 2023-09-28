import numpy as np
import pytest

sam = pytest.importorskip("ultrack.imgproc.sam")


def test_sam() -> None:
    image = np.random.rand(100, 100)
    seg_model = sam.MicroSAM()
    seg_model(image)

    # with maxima prompt
    seg_model = sam.set_peak_maxima_prompt(
        seg_model,
        sigma=5,
        min_distance=10,
        threshold_rel=0.1,
    )
    seg_model(image)
