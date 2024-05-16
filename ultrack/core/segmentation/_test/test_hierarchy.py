from typing import Tuple

import higra as hg
import numpy as np
import pytest
from skimage.segmentation import find_boundaries

from ultrack.core.segmentation.hierarchy import create_hierarchies
from ultrack.core.segmentation.vendored.hierarchy import to_labels


@pytest.mark.parametrize(
    "segmentation_mock_data",
    [{"size": 128, "n_dim": 2}, {"size": 64, "n_dim": 3}],
    indirect=True,
)
def test_horizontal_cut(
    segmentation_mock_data: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> None:
    foreground, edge, _ = segmentation_mock_data
    hierarchies = create_hierarchies(
        foreground,
        edge,
        hierarchy_fun=hg.watershed_hierarchy_by_dynamics,
        min_area=0,
        cut_threshold=0.1,
    )

    segmentation = to_labels(hierarchies, edge.shape)
    contours = find_boundaries(segmentation)

    # differente between contours
    difference = np.logical_xor(contours, edge).sum()

    # arbitrary proportion to take into account watershed tie-zoes
    assert difference < edge.size * 0.005
