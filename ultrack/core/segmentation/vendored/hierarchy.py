import logging
from functools import wraps
from typing import Any, Callable, List, Optional, Tuple

import higra as hg
import numpy as np
from numpy.typing import ArrayLike
from skimage import measure, segmentation
from skimage.measure._regionprops import RegionProperties

from ultrack.core.segmentation.vendored.graph import mask_to_graph
from ultrack.core.segmentation.vendored.node import Node

LOG = logging.getLogger(__name__)


def _cached(f: Callable) -> Callable:
    """Hierarchy data cache"""

    @wraps(f)
    def wrapper(obj: object) -> Any:
        name = f.__name__
        if name in obj._cache:
            return obj._cache[name]
        return f(obj)

    return wrapper


class Hierarchy:
    def __init__(
        self,
        props: RegionProperties,
        hierarchy_fun: Callable = hg.watershed_hierarchy_by_area,
        cut_threshold: Optional[float] = None,
        min_area: int = 50,
        max_area: int = 1000000,
        min_frontier: float = 0.0,
        anisotropy_pen: float = 0.0,
        cache: bool = True,
    ):
        """
        Helper class to interact with the watershed hierarchy of segments.

        Parameters
        ----------
        props : RegionProperties
            Region properties computed from scikit-image `measure.regionprops`.
        hierarchy_fun : Callable, optional
            Hierarchical watershed criterion function. Defaults to hg.watershed_hierarchy_by_area.
        cut_threshold : Optional[float], optional
            Hierarchy horizontal cut threshold. Defaults to None, returning a single labeling (no cut).
        min_area : int, optional
            Minimum number of pixels per segment. Defaults to 50.
        max_area : int, optional
            Maximum number of pixels per segment. Defaults to 1000000.
        min_frontier : float, optional
            Minimum required edge weight for splitting segments. Defaults to 0.0.
        anisotropy_pen : float, optional
            Edge weight penalization for the z-axis. Defaults to 0.0.
        cache : bool, optional
            Enables cache of data structures. Defaults to True.
            Can lead to segmentation fault when False if there's tie-zones on the watershed hierarchy.
            Due to tree and altitute miss match.
        """
        self.props = props
        self.hierarchy_fun = hierarchy_fun
        self.cut_threshold = cut_threshold
        self._min_area = min_area
        self._max_area = max_area
        self._min_frontier = min_frontier
        self._anisotropy_pen = anisotropy_pen

        self._cache = {}
        self._cache_active = cache

        self._mask_buffer: Optional[ArrayLike] = None
        self._tree_buffer: Optional[ArrayLike] = None

        self._nodes = {}

    def label_buffer(self, buffer: ArrayLike, offset: int = 1) -> int:
        """
        Fills the buffer with labels given an initial offset.

        Args:
            buffer (ArrayLike): Array to be filled, must match the original image shape.
            offset (int, optional): Labeling offset. Defaults to 1.

        Returns:
            int: Number of labeled segments.
        """
        if self.cut_threshold is None:
            buffer[self.props.slice][self.props.image] = offset
            return 1

        # cache must be on for cut
        cache_status = self.cache
        self.cache = True

        cut = self.cut.horizontal_cut_from_altitude(self.cut_threshold)
        relabeled, _, _ = segmentation.relabel_sequential(
            cut.labelisation_leaves(self.tree), offset=offset
        )
        buffer[self.props.slice][self.props.image] = relabeled

        self.cache = cache_status

        return len(cut.nodes())

    @property
    def cache(self) -> bool:
        return self._cache_active

    @cache.setter
    def cache(self, status: bool) -> None:
        self._cache_active = status
        self.props._cache_active = status
        if not status:
            hg.clear_auto_cache(reference_object=self.tree)
            self._cache.clear()
            self.props._cache.clear()
            self._nodes.clear()

    @staticmethod
    def _filter_contour_strength(
        tree: hg.Tree,
        alt: ArrayLike,
        frontier: ArrayLike,
        threshold: float,
        max_area: float,
    ) -> Tuple[hg.Tree, ArrayLike]:

        LOG.info("Filtering hierarchy by contour strength.")
        irrelevant_nodes = frontier < threshold

        if max_area is not None:
            # Avoid filtering nodes where merge leads to a node with maximum area above threshold
            parent_area = hg.attribute_area(tree)[tree.parents()]
            irrelevant_nodes[parent_area > max_area] = False

        tree, node_map = hg.simplify_tree(tree, irrelevant_nodes)
        return tree, alt[node_map], frontier[node_map]

    def watershed_hierarchy(self) -> Tuple[hg.Tree, ArrayLike, ArrayLike]:
        """
        Creates and filters the watershed hierarchy.

        Raises:
            RuntimeError: It returns a error to avoid SegFault from Higra
            when processing with a single node. Increase `min_area` to avoid this.

        Returns:
            Tuple[hg.Tree, ArrayLike]:
            It returns the watershed hierarchy (tree), its altitudes.
        """
        image = self.props.intensity_image
        if image.dtype == np.float16:
            image = image.astype(np.float32)

        if image.size < 8:
            raise RuntimeError(f"Region too small. Size of {image.size} found.")

        LOG.info("Creating graph from mask.")
        mask = self.props.image
        graph, weights = mask_to_graph(mask, image, self._anisotropy_pen)

        LOG.info("Constructing hierarchy.")
        tree, alt = self.hierarchy_fun(graph, weights)

        LOG.info("Filtering small nodes of hierarchy.")
        tree, alt = hg.filter_small_nodes_from_tree(tree, alt, self._min_area)

        hg.set_attribute(graph, "no_border_vertex_out_degree", None)
        frontier = hg.attribute_contour_strength(tree, weights)
        hg.set_attribute(graph, "no_border_vertex_out_degree", 2 * mask.ndim)

        if self._min_frontier > 0.0:
            tree, alt, frontier = self._filter_contour_strength(
                tree,
                alt,
                frontier,
                self._min_frontier,
                self._max_area,
            )

        LOG.info("Filtering large nodes of hierarchy.")
        tree, node_map = hg.simplify_tree(
            tree, hg.attribute_area(tree) > self._max_area
        )
        alt = alt[node_map]
        frontier = frontier[node_map]

        return tree, alt, frontier

    @property
    @_cached
    def area(self) -> ArrayLike:
        area = hg.attribute_area(self.tree)
        if self.cache:
            self._cache["area"] = area
        return area

    @property
    @_cached
    def dynamics(self) -> ArrayLike:
        dynamics = hg.attribute_dynamics(self.tree, self.alt)
        if self.cache:
            self._cache["dynamics"] = dynamics
        return dynamics

    @property
    @_cached
    def height(self) -> ArrayLike:
        height = hg.attribute_height(self.tree, self.alt)
        if self.cache:
            self._cache["height"] = height
        return height

    @property
    @_cached
    def tree(self) -> hg.Tree:
        tree, alt, frontier = self.watershed_hierarchy()
        if self.cache:
            self._cache["tree"] = tree
            self._cache["alt"] = alt
            self._cache["frontier"] = frontier
        return tree

    @property
    @_cached
    def alt(self) -> ArrayLike:
        tree, alt, frontier = self.watershed_hierarchy()
        if self.cache:
            self._cache["tree"] = tree
            self._cache["alt"] = alt
            self._cache["frontier"] = frontier
        return alt

    @property
    @_cached
    def frontier(self) -> hg.Tree:
        tree, alt, frontier = self.watershed_hierarchy()
        if self.cache:
            self._cache["tree"] = tree
            self._cache["alt"] = alt
            self._cache["frontier"] = frontier
        return frontier

    @property
    @_cached
    def cut(self) -> hg.HorizontalCutExplorer:
        cut = hg.HorizontalCutExplorer(self.tree, self.alt)
        if self.cache:
            self._cache["cut"] = cut
        return cut

    @staticmethod
    def create_node(node_index: int, parent: "Hierarchy", **kwargs) -> Node:
        """
        Helper function for creating nodes so it can be easily overwritten when inheriting
        the Hierarchy class.

        Args:
            node_index (int): Index of the node in the original hierarchy.
            parent (Hierarchy): Hierarchy containing the node.

        Returns:
            Node: Node object or a subclass of it.
        """
        return Node(node_index, parent=parent, **kwargs)

    def compute_nodes(self) -> None:
        """
        Compute the nodes from the hierarchy given the provided
        parameters (`min_area`, `max_area`, etc.)
        """
        tree = self.tree
        area = self.area
        frontier = self.frontier
        height = self.height

        for node_idx in tree.leaves_to_root_iterator(include_leaves=False):
            if area[node_idx] > self._max_area:
                continue
            self._nodes[node_idx] = self.create_node(
                node_idx,
                self,
                area=area[node_idx].item(),
                frontier=frontier[node_idx].item(),
                height=height[node_idx].item(),
            )

    def _fix_empty_nodes(self) -> None:
        """
        If no node is found due to the filtering operations,
        their root is added as a node to avoid an empty hierarchy.
        """
        if len(self._nodes) > 0:
            return

        root_index = self.tree.root()
        self._nodes[root_index] = self.create_node(
            root_index,
            self,
            area=self.props.area,
            frontier=-1.0,
            height=-1.0,
        )

    @property
    def nodes(self) -> List[Node]:
        if len(self._nodes) == 0:
            # avoiding multiple allocation during node computation
            self._mask_buffer = np.zeros(self.props.image.shape, dtype=int)
            self._tree_buffer = np.ones(self.tree.num_vertices(), dtype=bool)

            self.compute_nodes()
            self._fix_empty_nodes()

            self._mask_buffer = None
            self._tree_buffer = None

        return list(self._nodes.values())


def oversegment_components(
    labels: ArrayLike, boundaries: ArrayLike, max_area: int, anisotropy_pen: float = 0.0
) -> ArrayLike:
    """
    This function oversegment segments given an maximum area to decrease the overall hierarchy volume (area),
    speeding up the remaining computation.

    Parameters
    ----------
    labels : ArrayLike
        Input labels to be split into smaller segments.
    boundaries : ArrayLike
        Image graph edge weights.
    max_area : int
        Maximum area (hierarchy threshold cut).
    anisotropy_pen : float, optional
        Edge weight penalization for the z-axis. Defaults to 0.0.

    Returns
    -------
    ArrayLike
        Oversgmented labels given max area paremeter.
    """
    dtype = np.promote_types(np.float32, boundaries.dtype)
    boundaries = boundaries.astype(dtype)  # required by numba
    offset = 1
    new_labels = np.zeros_like(labels)
    for c in measure.regionprops(labels, boundaries):
        if c.area > max_area:
            graph, weights = mask_to_graph(c.image, c.intensity_image, anisotropy_pen)
            tree, alt = hg.watershed_hierarchy_by_area(graph, weights)
            cut = hg.labelisation_horizontal_cut_from_threshold(tree, alt, max_area)
            cut, _, _ = segmentation.relabel_sequential(cut, offset=offset)
            offset = cut.max() + 1
            new_labels[c.slice][c.image] = cut
        else:
            new_labels[c.slice][c.image] = offset
            offset += 1
    return new_labels


def to_labels(
    hierarchies: List[Hierarchy], shape: Tuple[int], dtype: np.dtype = np.int32
) -> None:
    labels = np.zeros(shape, dtype=dtype)

    offset = 1
    for h in hierarchies:
        offset += h.label_buffer(labels, offset=offset)

    return labels
