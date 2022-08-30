from enum import IntEnum

import numba
import numpy as np
from numba.experimental import jitclass


class Color(IntEnum):
    white = 0
    gray = 1
    black = 2


class Policy(IntEnum):
    minimum = 0
    maximum = 1


_SPEC = [
    ("values", numba.float32[:]),
    ("policy", numba.int8),
    ("_values", numba.float32[:]),
    ("_size", numba.int64),
    ("_age", numba.int64),
    ("_last", numba.int64),
    ("_policy", numba.int8),
    ("_nodes", numba.int64[:]),
    ("_pos", numba.int64[:]),
    ("_colors", numba.int8[:]),
    ("_ages", numba.int64[:]),
]


@jitclass(_SPEC)
class Heap:
    def __init__(self, values: np.ndarray, policy: Policy = Policy.maximum) -> None:
        """Argument heap, copied and modified from libift:
            https://github.com/PyIFT/pyift

        Parameters
        ----------
        values : np.ndarray
            Array of values used to heap sort, must be np.float32 (used by reference)
            for on-the-fly updates on the values.
        policy : Policy, optional
            Minimum or maximum policy, by default maximum.
        """
        size = len(values)
        self._values = values
        self._size = size
        self._age = 0
        self._last = -1
        self._policy = policy
        self._nodes = np.full(size, fill_value=-1, dtype=np.int64)
        self._pos = np.full(size, fill_value=-1, dtype=np.int64)
        self._colors = np.full(size, fill_value=Color.white, dtype=np.uint8)
        self._ages = np.zeros(size, dtype=np.int64)

    @staticmethod
    def _dad(i: int) -> int:
        return (i - 1) // 2

    @staticmethod
    def _left_son(i: int) -> int:
        return 2 * i + 1

    @staticmethod
    def _right_son(i: int) -> int:
        return 2 * i + 2

    def _swap(self, i: int, j: int) -> None:
        self._nodes[i], self._nodes[j] = self._nodes[j], self._nodes[i]
        self._pos[self._nodes[i]] = i
        self._pos[self._nodes[j]] = j

    def _lower(self, i: int, j: int) -> bool:
        ni, nj = self._nodes[i], self._nodes[j]
        if self._values[ni] == self._values[nj]:
            return self._ages[ni] < self._ages[nj]
        return self._values[ni] < self._values[nj]

    def _greater(self, i: int, j: int) -> bool:
        ni, nj = self._nodes[i], self._nodes[j]
        if self._values[ni] == self._values[nj]:
            return self._ages[ni] < self._ages[nj]  # always prioritize younger node
        return self._values[ni] > self._values[nj]

    def _go_up_heap_pos(self, pos: int) -> None:
        dad = self._dad(pos)
        if self._policy == Policy.minimum:
            while dad >= 0 and self._greater(dad, pos):
                self._swap(dad, pos)
                pos = dad
                dad = self._dad(pos)
        else:
            while dad >= 0 and self._lower(dad, pos):
                self._swap(dad, pos)
                pos = dad
                dad = self._dad(pos)

    def _go_down_heap_pos(self, pos: int) -> None:
        # numba wasn't compiling this function as a recursion
        # TODO: check if there's a better way to do this loop avoiding while True
        while True:
            next, left, right = pos, self._left_son(pos), self._right_son(pos)
            if self._policy == Policy.minimum:
                if left <= self._last and self._lower(left, next):
                    next = left
                if right <= self._last and self._lower(right, next):
                    next = right
            else:
                if left <= self._last and self._greater(left, next):
                    next = left
                if right <= self._last and self._greater(right, next):
                    next = right

            if next != pos:
                self._swap(next, pos)
                pos = next
            else:
                break

    def is_full(self) -> bool:
        return self._last == self._size - 1

    def is_empty(self) -> bool:
        return self._last == -1

    def insert(self, index: int) -> None:
        if self.is_full():
            raise ValueError("Heap is full.")

        # add node at last position
        self._last += 1
        self._age += 1
        self._nodes[self._last] = index
        self._ages[index] = self._age
        self._colors[index] = Color.gray
        self._pos[index] = self._last

        # sort heap
        self._go_up_heap_pos(self._last)

    def insert_array(self, array: np.ndarray) -> None:
        for i in array:
            self.insert(i)

    def pop(self) -> int:
        if self.is_empty():
            raise ValueError("Heap is empty.")

        # remove node at 0
        index = self._nodes[0]
        self._pos[index] = -1
        self._colors[index] = Color.black

        # swaps removed with last
        self._nodes[0] = self._nodes[self._last]
        self._pos[self._nodes[0]] = 0
        self._nodes[self._last] = -1
        self._last -= 1

        # sort heap
        self._go_down_heap_pos(0)

        return index

    def reset(self) -> None:
        self._nodes[...] = -1
        self._pos[...] = -1
        self._colors[...] = Color.white
        self._ages[...] = 0
        self._last = -1
        self._age = 0

    def go_up_heap(self, index: int) -> None:
        self._age += 1
        self._ages[index] = self._age
        self._go_up_heap_pos(self._pos[index])

    def go_down_heap(self, index: int) -> None:
        self._age += 1
        self._ages[index] = self._age
        self._go_down_heap_pos(self._pos[index])
