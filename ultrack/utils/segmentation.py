from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import scipy.ndimage as ndi
import zarr
from numpy.typing import ArrayLike
from tqdm import tqdm


@dataclass
class SegmentationChange:
    src_time_pt: int
    src_label: int
    dst_time_pt: int
    dst_label: int
    shift: ArrayLike
    bbox: Optional[ArrayLike] = None
    mask: Optional[ArrayLike] = None

    def src_mask_indices(self, include_time_pt: bool = False) -> Tuple[np.ndarray, ...]:
        """
        Get the indices of the source mask in the segmentation.

        Parameters
        ----------
        include_time_pt : bool
            If True, includes the time point in the indices.

        Returns
        -------
        Tuple[np.ndarray, ...]
            The indices of the source mask.
        """
        indices = list(np.nonzero(self.mask))
        for i, b in enumerate(self.bbox[: len(indices)]):
            indices[i] = np.round(indices[i] + b).astype(int)

        if include_time_pt:
            return (np.full_like(self.src_time_pt, len(indices[i])), *indices)
        else:
            return tuple(indices)

    def dst_mask_indices(self, include_time_pt: bool = None) -> Tuple[np.ndarray, ...]:
        """
        Get the indices of the destination mask in the segmentation.

        Parameters
        ----------
        include_time_pt : bool
            If True, includes the time point in the indices.

        Returns
        -------
        Tuple[np.ndarray, ...]
            The indices of the destination mask.
        """
        indices = list(self.src_mask_indices())
        for i, s in enumerate(self.shift):
            indices[i] = np.round(indices[i] + s).astype(int)

        if include_time_pt:
            return (np.full_like(indices[i], self.dst_time_pt), *indices)
        else:
            return tuple(indices)


class SegmentationPainter:
    """
    Class to apply in-place changes to a segmentation time lapse.

    Parameters
    ----------
    segments : ArrayLike
        The segmentation to be updated.
    """

    def __init__(self, segments: ArrayLike) -> None:
        self._segments = segments
        self._changes: Dict[int, List[SegmentationChange]] = defaultdict(list)

    def add_relabel(self, time_pt: int, src_label: int, dst_label: int) -> None:
        """
        Add a change in the segmentation at the given time point to the queue.

        Parameters
        ----------
        time_pt : int
            The time point at which the change occurred.
        src_label : int
            The previous label of the segment.
        dst_label : int
            The new label of the segment.
        """
        self._changes[time_pt].append(
            SegmentationChange(
                int(time_pt),
                int(src_label),
                int(time_pt),
                int(dst_label),
                np.zeros(self._segments.ndim - 1),
            ),
        )

    def add_new(
        self,
        src_time_pt: int,
        src_label: int,
        dst_time_pt: int,
        dst_label: int,
        shift: ArrayLike,
    ) -> None:
        """
        Add a new segment to the queue.

        Parameters
        ----------
        src_time_pt : int
            The time point of the reference segment.
        src_label : int
            The label of the reference segment.
        dst_time_pt : int
            The time point of the new segment.
        dst_label : int
            The label of the new segment.
        shift : ArrayLike
            The shift of the new segment.
        """
        self._changes[src_time_pt].append(
            SegmentationChange(
                int(src_time_pt),
                int(src_label),
                int(dst_time_pt),
                int(dst_label),
                shift,
            ),
        )

    def apply_changes(self) -> None:
        """
        Apply the changes to the segmentation.
        """
        for t, changes in tqdm(self._changes.items(), "Loading references masks"):
            frame = self._segments[t]
            label_to_slicing = {
                idx + 1: slicing
                for idx, slicing in enumerate(ndi.find_objects(frame))
                if slicing is not None
            }

            # first add to dict to avoid recomputing the mask
            label_to_bbox_mask = {}
            for change in changes:
                if change.src_label in label_to_slicing:
                    # creating cache of bbox and mask for the source label
                    if change.src_label not in label_to_bbox_mask:
                        slicing = label_to_slicing[change.src_label]
                        label_to_bbox_mask[change.src_label] = (
                            slicing,
                            frame[slicing] == change.src_label,
                        )
                    slicing, change.mask = label_to_bbox_mask[change.src_label]
                    change.bbox = np.asarray(
                        [s.start for s in slicing] + [s.stop for s in slicing]
                    )
                else:
                    raise ValueError(f"Label {change.src_label} not found in frame {t}")

        for t, changes in tqdm(self._changes.items(), "Applying changes"):
            for change in changes:
                dst_indices = change.dst_mask_indices(include_time_pt=True)
                if isinstance(self._segments, zarr.Array):
                    self._segments.vindex[dst_indices] = change.dst_label
                else:
                    self._segments[dst_indices] = change.dst_label

        self._changes.clear()
