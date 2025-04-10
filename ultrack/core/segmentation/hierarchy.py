import logging
from typing import Iterator, Tuple

import numpy as np
import scipy.ndimage as ndi
from numpy.typing import ArrayLike
from skimage import measure, morphology
from skimage.measure._regionprops import RegionProperties

from ultrack.core.segmentation.node import Node
from ultrack.core.segmentation.vendored.hierarchy import Hierarchy as _Hierarchy
from ultrack.core.segmentation.vendored.hierarchy import oversegment_components
from ultrack.utils.tiling import apply_tiled_and_stitch

LOG = logging.getLogger(__name__)


class Hierarchy(_Hierarchy):
    def __init__(self, props: RegionProperties, **kwargs) -> None:
        super().__init__(props, cache=False, **kwargs)
        self.bbox = props.bbox

    def free_props(self) -> None:
        """Frees region properties content."""
        del self.props
        self.props = None

    @staticmethod
    def create_node(node_index: int, parent: "Hierarchy", **kwargs) -> Node:
        """Overrides function to create nodes, see parent class documentation."""
        return Node(node_index, parent=parent, **kwargs)


def create_hierarchies(
    binary_foreground: ArrayLike,
    edge: ArrayLike,
    min_area_factor: float = 4.0,
    chunk_size: tuple[int, ...] = None,
    **kwargs,
) -> Iterator[Hierarchy]:
    """Computes a collection of hierarchical watersheds inside `binary_foreground` mask.

    Parameters
    ----------
    binary_foreground : ArrayLike
        Binary array showing regions of interest.

    edge : ArrayLike
        Fuzzy contour image representing instances boundaries.

    Returns
    -------
    Iterator[Hierarchy]
        List of hierarchical watersheds.
    """
    binary_foreground = np.asarray(binary_foreground)

    assert (
        issubclass(binary_foreground.dtype.type, np.integer)
        or binary_foreground.dtype == bool
    )

    LOG.info("Labeling connected components.")
    labels, num_labels = ndi.label(binary_foreground, output=np.int32)
    del binary_foreground

    if "min_area" in kwargs and num_labels > 1:
        LOG.info("Filtering small connected components.")
        #  To avoid removing small objects, divide by 4 to remove the smallest ones.
        #  Nodes in hierarchies are still filtered by minimum area.
        #  This is mainly for lonely cells.
        morphology.remove_small_objects(
            labels,
            min_size=int(kwargs["min_area"] / min_area_factor),
            out=labels,
        )

    edge = np.asarray(edge)

    if "max_area" in kwargs:
        LOG.info("Oversegmenting connected components.")

        if chunk_size is not None:

            def oversegment_tile(
                offset: int, *_args, **_kwargs
            ) -> Tuple[int, ArrayLike]:
                """Oversegment tile."""
                labels, offset = oversegment_components(
                    *_args,
                    **_kwargs,
                    max_area=kwargs["max_area"],
                    anisotropy_pen=kwargs.get("anisotropy_pen", 0.0),
                    offset=offset,
                )
                return offset, labels

            apply_tiled_and_stitch(
                labels,
                edge,
                func=oversegment_tile,
                chunk_size=chunk_size,
                out_array=labels,
            )

        # executed twice for chunked processing to process merged regions
        labels, _ = oversegment_components(
            labels,
            edge,
            max_area=kwargs["max_area"],
            anisotropy_pen=kwargs.get("anisotropy_pen", 0.0),
        )

    LOG.info("Creating hierarchies (lazy).")
    for c in measure.regionprops(labels, edge, cache=True):
        yield Hierarchy(c, **kwargs)
