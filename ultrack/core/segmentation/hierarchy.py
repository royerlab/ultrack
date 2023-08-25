import logging
from typing import List

import numpy as np
from numpy.typing import ArrayLike
from skimage import measure, morphology
from skimage.measure._regionprops import RegionProperties

from ultrack.core.segmentation.node import Node
from ultrack.core.segmentation.vendored.hierarchy import Hierarchy as _Hierarchy
from ultrack.core.segmentation.vendored.hierarchy import oversegment_components

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
    binary_detection: ArrayLike,
    edge: ArrayLike,
    **kwargs,
) -> List[Hierarchy]:
    """Computes a collection of hierarchical watersheds inside `binary_detection` mask.

    Parameters
    ----------
    binary_detection : ArrayLike
        Binary array showing regions of interest.

    edge : ArrayLike
        Fuzzy contour image representing instances boundaries.

    Returns
    -------
    List[Hierarchy]
        List of hierarchical watersheds.
    """
    binary_detection = np.asarray(binary_detection)
    edge = np.asarray(edge)

    assert (
        issubclass(binary_detection.dtype.type, np.integer)
        or binary_detection.dtype == bool
    )

    LOG.info("Labeling connected components.")
    labels, num_labels = measure.label(
        binary_detection, return_num=True, connectivity=1
    )
    labels = labels.astype(np.int32)

    if "min_area" in kwargs and num_labels > 1:
        LOG.info("Filtering small connected components.")
        labels = morphology.remove_small_objects(labels, min_size=kwargs["min_area"])

    if "max_area" in kwargs:
        LOG.info("Oversegmenting connected components.")
        labels = oversegment_components(
            labels,
            edge,
            kwargs["max_area"],
            kwargs.get("anisotropy_pen", 0.0),
        )

    LOG.info("Creating hierarchies (lazy).")
    return [
        Hierarchy(c, **kwargs) for c in measure.regionprops(labels, edge, cache=True)
    ]
