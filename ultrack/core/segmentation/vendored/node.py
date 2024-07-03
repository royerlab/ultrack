import itertools
from typing import TYPE_CHECKING, Optional, Tuple

import higra as hg
import numpy as np
from numpy.typing import ArrayLike

if TYPE_CHECKING:
    from ultrack.core.segmentation.vendored.hierarchy import Hierarchy


class Node:
    def __init__(
        self, h_node_index: int, parent: Optional["Hierarchy"], **kwargs
    ):  # noqa

        for key, value in kwargs.items():
            setattr(self, key, value)

        self._h_node_index = h_node_index
        self._parent = parent
        if parent is not None:
            self.bbox, self.mask = self._init_bbox_and_mask()
        else:
            self.bbox, self.mask = None, None

    @staticmethod
    def _fast_find_binary_object(mask: ArrayLike) -> Tuple[slice]:
        """
        Fast bounding box algorithm from: https://stackoverflow.com/questions/31400769/bounding-box-of-numpy-array

        Args:
            mask (ArrayLike): Binary mask.

        Returns:
            Tuple[slice]: Bounding-box.
        """

        ndim = mask.ndim
        slicing = []
        for ax in itertools.combinations(reversed(range(ndim)), ndim - 1):
            mask_slice = np.any(mask, axis=ax)
            nonzeros = mask_slice.nonzero()[0]
            slicing.append(slice(nonzeros[0], nonzeros[-1] + 1))
        return tuple(slicing)

    def _reduce_mask(self, mask: ArrayLike) -> Tuple[ArrayLike, ArrayLike]:
        slicing = self._fast_find_binary_object(mask)
        crop_slicing = self._parent.props.slice
        bbox = np.array(
            [crop_slicing[i].start + slicing[i].start for i in range(mask.ndim)]
            + [
                crop_slicing[i].start + slicing[i].stop for i in range(mask.ndim)
            ]  # noqa
        )

        return bbox, mask[slicing].copy()

    def _bbox_and_mask_from_leaves(self) -> Tuple[ArrayLike, ArrayLike]:
        crop_mask = self._parent.props.image
        tree = self._parent.tree

        size = tree.num_vertices()
        if self._parent._tree_buffer is None:
            not_selected = np.ones(size, dtype=bool)
        else:
            not_selected = self._parent._tree_buffer

        not_selected[self._h_node_index] = False
        leaves_labels = hg.reconstruct_leaf_data(tree, np.arange(size), not_selected)
        not_selected[self._h_node_index] = True  # resetting

        if self._parent._mask_buffer is None:
            mask = np.zeros(crop_mask.shape, dtype=int)
        else:
            mask = self._parent._mask_buffer

        mask[crop_mask] = leaves_labels
        binary_mask = mask == self._h_node_index
        mask[crop_mask] = 0  # resetting

        return self._reduce_mask(binary_mask)

    def _bbox_and_mask_from_non_leaves(
        self, children: ArrayLike, tree: Optional[hg.Tree] = None
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Computes bounding-box and mask of non-leaf nodes using the
        previously data from its children. Much faster than computing from the hierarchy leaves.

        Args:
            children (ArrayLike): Array of non-leaf children nodes
            tree (Optional[hg.Tree], optional): Pre-computed hierarchy tree. Defaults to None.

        Returns:
            Tuple[ArrayLike, ArrayLike]: Bounding-box and mask.
        """
        if tree is None:
            tree = self._parent.tree

        ndim = self._parent.props._ndim
        bbox = np.zeros(2 * ndim, dtype=int)
        bbox[:ndim] = np.iinfo(int).max

        for child in children:
            try:
                # this can happen due to the maximum size selecion
                child_bbox = self._parent._nodes[child].bbox
            except KeyError:
                return self._bbox_and_mask_from_leaves()

            for i in range(ndim):
                bbox[i] = min(bbox[i], child_bbox[i])
                bbox[i + ndim] = max(bbox[i + ndim], child_bbox[i + ndim])

        shape = tuple(M - m for m, M in zip(bbox[:ndim], bbox[ndim:]))
        mask = np.zeros(shape, dtype=bool)
        for child in children:
            child_bbox = self._parent._nodes[child].bbox
            slicing = []
            for i in range(ndim):
                start = child_bbox[i] - bbox[i]
                end = start + child_bbox[i + ndim] - child_bbox[i]
                slicing.append(slice(start, end))
            slicing = tuple(slicing)
            mask[slicing] |= self._parent._nodes[child].mask

        return bbox, mask

    def _init_bbox_and_mask(self) -> Tuple[ArrayLike, ArrayLike]:
        tree = self._parent.tree

        children = tree.children(self._h_node_index)
        num_leaf_children = (children < tree.num_leaves()).sum()

        if num_leaf_children > 0:
            return self._bbox_and_mask_from_leaves()

        return self._bbox_and_mask_from_non_leaves(
            children[children >= tree.num_leaves()], tree
        )
