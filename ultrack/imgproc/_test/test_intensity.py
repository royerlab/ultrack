import numpy as np

from ultrack.imgproc import normalize


def test_basic_normalization() -> None:
    image = np.array([[[[-0.2, 0.5], [0.4, 0.8]], [[0.6, 0.3], [1.1, 0.9]]]])
    normalized = normalize(image, 1.0)

    assert np.all(normalized >= 0) and np.all(normalized <= 1)
    assert not np.array_equal(image, normalized)
