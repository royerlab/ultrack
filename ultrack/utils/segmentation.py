from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import scipy.ndimage as ndi
import zarr
from numpy.typing import ArrayLike
from tqdm import tqdm
from zarr.storage import Store

from ultrack.utils.array import create_zarr


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
            indices[i] = (indices[i] + b).astype(int)

        if include_time_pt:
            return (np.full_like(self.src_time_pt, len(indices[i])), *indices)
        else:
            return tuple(indices)

    def dst_mask_indices(self, include_time_pt: bool = False) -> Tuple[np.ndarray, ...]:
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
            indices[i] = (indices[i] + s).astype(int)

        if include_time_pt:
            return (np.full_like(indices[i], self.dst_time_pt), *indices)
        else:
            return tuple(indices)

    def __str__(self) -> str:
        return (
            "Segmentation update\n"
            f"from src_time_pt={self.src_time_pt} to dst_time_pt={self.dst_time_pt}\n"
            f"from src_label={self.src_label} to dst_label={self.dst_label}\n"
            f"with shift {self.shift}"
        )

    def __repr__(self) -> str:
        return str(self)


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
        self.src_tp_to_changes: Dict[int, List[SegmentationChange]] = defaultdict(list)
        self.dst_tp_to_changes: Dict[int, List[SegmentationChange]] = defaultdict(list)
        self._remap: Dict[Tuple[int, int], int] = {}

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
        time_pt = int(time_pt)
        src_label = int(src_label)
        dst_label = int(dst_label)

        # check if it's accumulating changes
        for change in self.dst_tp_to_changes[time_pt]:
            # accumulate changes and returns without adding
            # this could happen in consecutive changes to the same label
            if change.dst_label == src_label:
                change.dst_label = dst_label
                return

        segm_change = SegmentationChange(
            time_pt,
            src_label,
            time_pt,
            dst_label,
            np.zeros(self._segments.ndim - 1),
        )

        self.src_tp_to_changes[time_pt].append(segm_change)
        self.dst_tp_to_changes[time_pt].append(segm_change)

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
        segm_change = SegmentationChange(
            int(src_time_pt),
            int(src_label),
            int(dst_time_pt),
            int(dst_label),
            shift,
        )

        self.src_tp_to_changes[src_time_pt].append(segm_change)
        self.dst_tp_to_changes[dst_time_pt].append(segm_change)

    def apply_changes(self) -> None:
        """
        Apply the changes to the segmentation.

        It's optimized to avoid reloading the same time point multiple times.
        """

        # Loading source time points masks
        for t, changes in tqdm(
            self.src_tp_to_changes.items(), "Loading references masks"
        ):
            frame = np.asarray(self._segments[t])
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
                    if frame.sum() == 0:
                        raise ValueError(f"Frame {t} is empty. Something went wrong.")

                    print(f"{change} not found in frame {t}")

        # Applying changes to destination
        for t, changes in tqdm(self.dst_tp_to_changes.items(), "Applying changes"):
            frame = np.asarray(self._segments[t])
            for change in changes:
                if change.mask is None:
                    print(f"Mask not found for {change}")
                    continue
                dst_indices = change.dst_mask_indices()
                dst_indices = tuple(
                    np.maximum(0, np.minimum(i, s - 1))
                    for i, s in zip(dst_indices, frame.shape)
                )  # clipping mask to frame
                frame[dst_indices] = change.dst_label
            self._segments[t] = frame

        self.src_tp_to_changes.clear()
        self.dst_tp_to_changes.clear()


def copy_segments(
    segments: ArrayLike,
    segments_store_or_path: Union[Store, Path, str, None] = None,
    overwrite: bool = False,
) -> zarr.Array:
    """
    Copy the segments to a new zarr array.

    Parameters
    ----------
    segments : ArrayLike
        The segments to copy.
    segments_store_or_path : Union[Store, Path, str, None]
        The store or path to save the new segments.
    overwrite : bool
        If True, overwrite the existing array.

    Returns
    -------
    zarr.Array
        The new segments array.
    """
    is_zarr = isinstance(segments, zarr.Array)
    out_segments = create_zarr(
        segments.shape,
        segments.dtype,
        segments_store_or_path,
        chunks=segments.chunks if is_zarr else None,
        overwrite=overwrite,
    )

    if is_zarr:
        print("Copying segments...")
        # not very clean because we just created the array above
        zarr.copy_store(segments.store, out_segments.store, if_exists="replace")
        out_segments = zarr.open(
            out_segments.store
        )  # not sure why this is necessary in large datasets
    else:
        for t in tqdm(range(segments.shape[0]), "Copying segments"):
            out_segments[t] = segments[t]

    return out_segments
