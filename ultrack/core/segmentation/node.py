import logging
from typing import Dict, Optional, Sequence, Tuple, Union

import blosc2
import numpy as np
import scipy.ndimage as ndi
import zarr
from numba import njit, types
from numpy.typing import ArrayLike
from scipy import fft

from ultrack.core.segmentation.vendored.node import Node as _Node

try:
    import tensorstore as ts
except ImportError:
    ts = None


LOG = logging.getLogger(__name__)


@njit
def intersects(bbox1: types.int64[:], bbox2: types.int64[:]) -> bool:
    """Checks if bounding box intersects by checking if their coordinates are within the
    range of the other along each axis.

    Parameters
    ----------
    bbox1 : ArrayLike
        Bounding box as (min_0, min_1, ..., max_0, max_1, ...).
    bbox2 : ArrayLike
        Bounding box as (min_0, min_1, ..., max_0, max_1, ...).
    Returns
    -------
    bool
        Boolean indicating intersection.
    """
    n_dim = len(bbox1) // 2
    intersects = True
    for i in range(n_dim):
        k = i + n_dim
        intersects &= (
            bbox1[i] <= bbox2[i] < bbox1[k]
            or bbox1[i] < bbox2[k] <= bbox1[k]
            or bbox2[i] <= bbox1[i] < bbox2[k]
            or bbox2[i] < bbox1[k] <= bbox2[k]
        )
        if not intersects:
            break

    return intersects


@njit
def iou_with_bbox_2d(
    bbox1: np.ndarray, bbox2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray
) -> float:
    y_min = max(bbox1[0], bbox2[0])
    x_min = max(bbox1[1], bbox2[1])
    y_max = min(bbox1[2], bbox2[2])
    x_max = min(bbox1[3], bbox2[3])

    aligned_mask1 = mask1[
        y_min - bbox1[0] : mask1.shape[0] + y_max - bbox1[2],
        x_min - bbox1[1] : mask1.shape[1] + x_max - bbox1[3],
    ]

    aligned_mask2 = mask2[
        y_min - bbox2[0] : mask2.shape[0] + y_max - bbox2[2],
        x_min - bbox2[1] : mask2.shape[1] + x_max - bbox2[3],
    ]

    inter = np.logical_and(aligned_mask1, aligned_mask2).sum()
    if inter == 0:  # this avoids dividing by zero when the bbox don't intersect
        return 0
    union = mask1.sum() + mask2.sum() - inter
    return (inter / union).item()


@njit
def iou_with_bbox_3d(
    bbox1: np.ndarray, bbox2: np.ndarray, mask1: np.ndarray, mask2: np.ndarray
) -> float:
    z_min = max(bbox1[0], bbox2[0])
    y_min = max(bbox1[1], bbox2[1])
    x_min = max(bbox1[2], bbox2[2])
    z_max = min(bbox1[3], bbox2[3])
    y_max = min(bbox1[4], bbox2[4])
    x_max = min(bbox1[5], bbox2[5])

    aligned_mask1 = mask1[
        z_min - bbox1[0] : mask1.shape[0] + z_max - bbox1[3],
        y_min - bbox1[1] : mask1.shape[1] + y_max - bbox1[4],
        x_min - bbox1[2] : mask1.shape[2] + x_max - bbox1[5],
    ]

    aligned_mask2 = mask2[
        z_min - bbox2[0] : mask2.shape[0] + z_max - bbox2[3],
        y_min - bbox2[1] : mask2.shape[1] + y_max - bbox2[4],
        x_min - bbox2[2] : mask2.shape[2] + x_max - bbox2[5],
    ]

    inter = np.logical_and(aligned_mask1, aligned_mask2).sum()
    if inter == 0:  # this avoids dividing by zero when the bbox don't intersect
        return 0
    union = mask1.sum() + mask2.sum() - inter
    return (inter / union).item()


class Node(_Node):
    def __init__(self, h_node_index: int, id: int = -1, time: int = -1, **kwargs):
        self.area = None
        super().__init__(h_node_index=h_node_index, **kwargs)
        self.id = id
        self.time = time
        if self.mask is None:
            self.centroid = None
        else:
            self.centroid = self._centroid()

        LOG.info("Constructed node %s", self)

    def IoU(self, other: "Node") -> float:
        if not intersects(self.bbox, other.bbox):
            return 0.0
        if self.mask.ndim == 2:
            return iou_with_bbox_2d(self.bbox, other.bbox, self.mask, other.mask)
        elif self.mask.ndim == 3:
            return iou_with_bbox_3d(self.bbox, other.bbox, self.mask, other.mask)
        else:
            raise NotImplementedError

    def precompute_dct(
        self,
        images: Sequence[ArrayLike],
        size: int = 10,
    ) -> None:
        if len(images) == 1:
            self.dct = self._gray_dct(images[0], size)

        else:
            self.dct = np.stack([self._gray_dct(image, size) for image in images])
        self.dct /= np.linalg.norm(self.dct)

    def _gray_dct(self, image: ArrayLike, size: int) -> None:
        crop = self.roi(image)
        dct = fft.dctn(crop)[(slice(None, size),) * crop.ndim]
        shape_diff = size - np.array(dct.shape)
        if np.any(shape_diff > 0):
            dct = np.pad(dct, tuple((0, max(0, s)) for s in shape_diff))
        return dct

    def dct_dot(self, other: "Node") -> float:
        return np.sum(self.dct * other.dct)

    def distance(self, other: "Node") -> float:
        return np.sqrt(np.square(self.centroid - other.centroid).sum())

    def mask_indices(self) -> Tuple[np.ndarray]:
        indices = list(np.nonzero(self.mask))
        for i in range(len(indices)):
            indices[i] += self.bbox[i]  # centering at bbox
        return tuple(indices)

    def contains(self, coords: Union[np.ndarray, Tuple]) -> bool:
        coords = np.round(coords)
        indices = np.asarray(self.mask_indices()).T
        dist = np.square(indices - coords).sum(axis=1)
        return np.any(dist < 1)

    @property
    def slice(self) -> Tuple[slice, slice, slice]:
        bbox = self.bbox
        ndim = self.mask.ndim
        return tuple(slice(bbox[i], bbox[i + ndim]) for i in range(ndim))

    def roi(self, image: Union[zarr.Array, np.ndarray]) -> np.ndarray:
        return np.asarray(image[self.slice])

    def masked_roi(self, image: Union[zarr.Array, np.ndarray]) -> np.ndarray:
        crop = image[self.slice].copy()
        crop[np.logical_not(self.mask)] = 0
        return crop

    def paint_buffer(
        self, buffer: ArrayLike, value: Optional[int] = None, include_time: bool = True
    ) -> None:
        if value is None:
            value = self.id

        indices = self.mask_indices()
        if include_time:
            indices = (self.time,) + indices

        if isinstance(buffer, zarr.Array) or (
            ts is not None and isinstance(buffer, ts.TensorStore)
        ):
            buffer.vindex[indices] = value
        else:
            buffer[indices] = value

    def check_bbox_and_mask(self) -> None:
        shape = self.bbox[self.mask.ndim :] - self.bbox[: self.mask.ndim]
        assert np.all(
            shape == self.mask.shape
        ), f"mask: {self.mask.shape}, bbox: {shape}"

    def _centroid(self) -> np.ndarray:
        centroid = (
            np.array(np.nonzero(self.mask)).mean(axis=1) + self.bbox[: self.mask.ndim]
        )
        return centroid.round().astype(int)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        return self.id == other.id and self._h_node_index == other._h_node_index

    def __lt__(self, other: "Node") -> bool:
        return self.id < other.id

    def __gt__(self, other: "Node") -> bool:
        return self.id > other.id

    def __hash__(self) -> int:
        return hash((self.id, self._h_node_index))

    def same(self, other: "Node") -> bool:
        if self.area != other.area or self.mask.shape != other.mask.shape:
            return False

        return (
            np.all(self.centroid == other.centroid)
            and np.all(self.bbox == other.bbox)
            and np.all(self.mask == other.mask)
        )

    def __repr__(self) -> str:
        return self.__str__()

    def __str__(self) -> str:
        return (
            f"index: {self._h_node_index}\n"
            + f"bbox: {self.bbox}\n"
            + f"area: {self.area}\n"
            + f"centroid: {self.centroid}\n"
        )

    def __getstate__(self) -> Dict:
        d = self.__dict__.copy()
        d["_parent"] = None
        d["mask"] = blosc2.pack_array(self.mask)
        return d

    def __setstate__(self, d: Dict) -> None:
        d["mask"] = blosc2.unpack_array(d["mask"])
        self.__dict__ = d

    def intensity_mean(
        self,
        image: ArrayLike,
    ) -> ArrayLike:
        """Compute the mean intensity feature for this node."""
        features = np.mean(image[self.mask_indices()], axis=0)
        assert features.shape[0] == image.shape[-1], f"{features.shape}, {image.shape}"
        return features

    def intensity_std(
        self,
        image: ArrayLike,
    ) -> ArrayLike:
        """Compute the standard deviation of intensity feature for this node."""
        features = image[self.mask_indices()].std(axis=0)
        assert features.shape[0] == image.shape[-1], f"{features.shape}, {image.shape}"
        return features

    @staticmethod
    def from_mask(
        time: int,
        mask: ArrayLike,
        bbox: Optional[ArrayLike] = None,
        node_id: int = -1,
    ) -> "Node":
        """
        Create a new node from a mask.

        Parameters
        ----------
        time : int
            Time point of the node.
        mask : ArrayLike
            Binary mask of the node.
        bbox : Optional[ArrayLike], optional
            Bounding box of the node, (min_0, min_1, ..., max_0, max_1, ...).
            When provided it assumes the mask is a crop of the original image, by default None
        node_id : int, optional
            Node ID, by default -1

        Returns
        -------
        Node
            New node.
        """

        if mask.dtype != bool:
            raise ValueError(f"Mask should be a boolean array. Found {mask.dtype}")

        node = Node(h_node_index=-1, id=node_id, time=time, parent=None)

        if bbox is None:
            bbox = ndi.find_objects(mask)[0]
            mask = mask[bbox]

        bbox = np.asarray(bbox)

        if mask.ndim * 2 != len(bbox):
            raise ValueError(
                f"Bounding box {bbox} does not match 2x mask ndim {mask.ndim}"
            )

        bbox_shape = bbox[mask.ndim :] - bbox[: mask.ndim]

        if np.any(bbox_shape != mask.shape):
            raise ValueError(
                f"Mask shape {mask.shape} does not match bbox shape {bbox_shape}"
            )

        node.bbox = bbox
        node.mask = mask
        node.area = mask.sum()
        node.centroid = node._centroid()

        return node
