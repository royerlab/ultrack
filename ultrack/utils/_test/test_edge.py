from typing import List

import numpy as np
import pytest

from ultrack.utils.cuda import to_cpu
from ultrack.utils.data import make_segmentation_mock_data
from ultrack.utils.edge import labels_to_contours


@pytest.fixture
def multiple_labels(
    n_samples: int = 3,
    length: int = 2,
    size: int = 128,
    n_dim: int = 3,
) -> List[np.ndarray]:
    """Creates a list of sequences of labels."""

    rng = np.random.default_rng(42)
    labels = []

    for _ in range(n_samples):
        lb = np.empty((length,) + (size,) * n_dim, dtype=int)
        for t in range(length):
            _, _, lb[t] = make_segmentation_mock_data(size=size, n_dim=n_dim, rng=rng)
        labels.append(lb)

    return labels


def test_labels_to_contours(multiple_labels: List[np.ndarray]) -> None:
    """Tests merge and convertion of multiple labels into foreground and contours."""

    foreground, _ = to_cpu(labels_to_contours(multiple_labels, sigma=1.5))

    shape = multiple_labels[0].shape

    for lb in multiple_labels:
        for t in range(shape[0]):
            mask = lb[t] > 0
            assert np.all(foreground[t][mask] > 0)
