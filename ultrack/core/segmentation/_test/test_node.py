from typing import List

import numpy as np
import pytest

from ultrack.core.segmentation.node import intersects


@pytest.mark.parametrize(
    "bbox1,bbox2,solution",
    [
        ((0, 0, 5, 5), (1, 1, 2, 2), True),
        ((1, 1, 2, 2), (0, 0, 5, 5), True),
        ((2, 2, 2, 6, 6, 6), (1, 1, 1, 4, 4, 4), True),
        ((5, 5, 7, 7), (3, 2, 6, 4), False),
    ],
)
def test_bbox_intersection(bbox1: List[int], bbox2: List[int], solution: bool) -> None:
    bbox1, bbox2 = np.asarray(bbox1), np.asarray(bbox2)
    assert intersects(bbox1, bbox2) == solution
