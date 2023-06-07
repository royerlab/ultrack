import numpy as np
import pytest

from ultrack.core.solve.solver.heuristic._heap import Heap, Policy


@pytest.mark.parametrize(
    "policy",
    [Policy.minimum, Policy.maximum],
)
def test_heap(policy: Policy) -> None:
    size = 10_000
    rng = np.random.default_rng(42)

    values = rng.uniform(size=size).astype(np.float32)
    heap = Heap(values, policy=policy)

    if policy == Policy.minimum:
        nil_value = np.finfo(np.float32).max
        arg_func = np.argmin
    else:
        nil_value = np.finfo(np.float32).min
        arg_func = np.argmax

    count = 0
    # testing partial insertion with removal
    for i in range(size):
        heap.insert(i)
        if (i + 1) % 5 == 0:
            j = heap.pop()
            assert j == arg_func(values[: i + 1])
            values[j] = nil_value
            count += 1

    # testing removal until it's empty.
    while not heap.is_empty():
        i = heap.pop()
        assert i == arg_func(values)
        values[i] = nil_value
        count += 1

    # asserting the right number of elements where removed
    assert count == size


def test_insert_array() -> None:
    size = 1_000
    rng = np.random.default_rng(42)

    values = rng.uniform(size=size).astype(np.float32)
    heap = Heap(values, policy=Policy.minimum)

    heap.insert_array(np.arange(size))
    while not heap.is_empty():
        i = heap.pop()
        assert i == np.argmin(values)
        values[i] = np.finfo(np.float32).max
