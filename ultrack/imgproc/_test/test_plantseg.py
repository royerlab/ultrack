from typing import Tuple, Union

import numpy as np
import pytest

from ultrack.imgproc.plantseg import PlantSeg


@pytest.mark.parametrize(
    "transpose",
    [
        None,
        (2, 1, 0),
    ],
)
def test_plantseg(
    transpose: Union[None, Tuple[int, int, int]],
) -> None:

    image = np.random.rand(100, 100, 100)

    try:
        seg_model = PlantSeg(
            model_name="generic_light_sheet_3D_unet",
            batch_size=1,
            patch=image.shape,
        )
    except ModuleNotFoundError:
        pytest.skip("PlantSeg not installed")

    seg_model(image, transpose=transpose)
