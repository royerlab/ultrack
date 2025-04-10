from typing import Callable, Tuple

import numpy as np
from numba import njit
from numpy.typing import ArrayLike
from scipy import sparse
from tqdm import tqdm


@njit
def _stitch_tiled_arrays_2d(
    arr: ArrayLike,
    chunk_size: Tuple[int, ...],
    num_chunks: Tuple[int, ...],
) -> set[tuple[int, int]]:

    adj = set()

    for c_y in range(num_chunks[0]):
        for c_x in range(num_chunks[1]):

            # iterate over y boundary
            if c_x < num_chunks[1] - 1:
                x_right = (c_x + 1) * chunk_size[1]
                x_left = x_right - 1

                for y in range(
                    c_y * chunk_size[0], min((c_y + 1) * chunk_size[0], arr.shape[0])
                ):
                    left_value = arr[y, x_left]
                    right_value = arr[y, x_right]
                    if left_value != 0 and right_value != 0:
                        adj.add((left_value, right_value))

            # iterate over x boundary
            if c_y < num_chunks[0] - 1:
                y_top = (c_y + 1) * chunk_size[0]
                y_bottom = y_top - 1

                for x in range(
                    c_x * chunk_size[1], min((c_x + 1) * chunk_size[1], arr.shape[1])
                ):
                    top_value = arr[y_top, x]
                    bottom_value = arr[y_bottom, x]
                    if top_value != 0 and bottom_value != 0:
                        adj.add((top_value, bottom_value))
    return adj


def apply_tiled_and_stitch(
    *in_arrays: ArrayLike,
    func: Callable[[int, ArrayLike], tuple[int, ArrayLike]],
    chunk_size: Tuple[int, ...],
    out_array: ArrayLike,
) -> None:
    """
    Apply a function to tiled chunks of input arrays and stitch the results together.
    Stitching is done by remapping the values of the regions on the edges of the tiles
    so they match.

    'func' example, for connected component labeling:

    >>> def cc_labeling(offset: int, arr: np.ndarray) -> tuple[int, np.ndarray]:
    >>>     offset = max(1, offset)
    >>>     out, _ = ndi.label(arr)
    >>>     out = relabel_sequential(out, offset=offset)[0]
    >>>     return max(out.max(), offset) + 1, out

    Parameters
    ----------
    in_arrays : ArrayLike
        Input arrays to apply the function to.
    func : Callable[[int, ArrayLike], tuple[ArrayLike, int]]
        2D or 3D function to apply to the input arrays.
        The function should take an offset and an array as input and return a tuple
        with the next offset and the resulting array.
    chunk_size : Tuple[int, ...]
        Size of the chunks to create for tiling.
    out_array : ArrayLike
        Output array to store the results.
    """
    name = func.__name__ if hasattr(func, "__name__") else type(func).__name__

    if len(chunk_size) != 2 and len(chunk_size) != 3:
        raise ValueError(
            f"Chunk size must be 2 or 3 dimensions, got '{len(chunk_size)}'."
        )

    for arr in in_arrays:
        if arr.shape != out_array.shape:
            raise ValueError(
                f"Input array shape '{arr.shape}' does not match output array shape '{out_array.shape}'."
            )
        if len(chunk_size) != arr.ndim:
            raise ValueError(
                f"Chunk size '{chunk_size}' does not match input array ndim '{arr.ndim}'."
            )

    chunk_size = np.minimum(chunk_size, out_array.shape)

    # Calculate the number of chunks in each dimension
    num_chunks = tuple(
        int(np.ceil(out_array.shape[i] / chunk_size[i])) for i in range(out_array.ndim)
    )

    offset = 1
    for start_indices in tqdm(list(np.ndindex(*num_chunks)), desc=f"Applying '{name}'"):
        slicing = tuple(
            slice(
                start_indices[i] * chunk_size[i],
                min((start_indices[i] + 1) * chunk_size[i], out_array.shape[i]),
            )
            for i in range(out_array.ndim)
        )
        next_offset, out_array[slicing] = func(
            offset, *[arr[slicing] for arr in in_arrays]
        )
        if next_offset < offset:
            raise ValueError(
                f"Provided function '{name}' must return a new offset greater than "
                "the previous offset. Labeling won't work properly."
            )
        offset = next_offset

    if out_array.ndim == 2:
        adj = _stitch_tiled_arrays_2d(out_array, chunk_size, num_chunks)
    elif out_array.ndim == 3:
        # create 3d adjacency graph by evaluating in orthogonal 2d planes
        adj = set()
        for z in range(out_array.shape[0]):
            adj |= _stitch_tiled_arrays_2d(out_array[z], chunk_size[1:], num_chunks[1:])
        for y in range(out_array.shape[1]):
            adj |= _stitch_tiled_arrays_2d(
                out_array[:, y],
                (chunk_size[0], chunk_size[2]),
                (num_chunks[0], num_chunks[2]),
            )
    else:
        ValueError(f"Unsupported number of dimensions: {out_array.ndim}.")

    if len(adj) == 0:
        return

    row, col = np.asarray(list(adj)).T
    n_labels = out_array.max() + 1
    adj_mat = sparse.csr_array(
        (np.ones(len(row), dtype=bool), (row, col)),
        shape=(n_labels, n_labels),
    )
    _, cc = sparse.csgraph.connected_components(adj_mat, directed=True)

    cc = cc.astype(out_array.dtype)

    if cc[0] != 0:  # mapping background (0) to 0 if not already
        cc += 1
        cc[0] = 0

    out_array[...] = cc[out_array]
