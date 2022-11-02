from typing import Callable, List, Optional, Tuple

import higra as hg
import numpy as np
from numpy.typing import ArrayLike
from skimage import measure, morphology, segmentation
from skimage.measure._regionprops import RegionProperties

from ultrack.core.segmentation.vendored.graph import mask_to_graph
from ultrack.core.segmentation.vendored.node import Node


class Hierarchy:
    def __init__(
        self,
        props: RegionProperties,
        hierarchy_fun: Callable = hg.watershed_hierarchy_by_area,
        cut_threshold: Optional[float] = None,
        min_area: int = 50,
        max_area: int = 1000000,
        min_frontier: float = 0.25,
        anisotropy_pen: float = 0.0,
        cache: bool = False,
    ):
        """
        Helper class to interact with the watershed hierarchy of segments.

        Args:
            props (RegionProperties): Region properties computed from scikit-image `measure.regionprops`.
            hierarchy_fun (Callable, optional):
                Hierarchical watershed criterion function. Defaults to hg.watershed_hierarchy_by_area.
            cut_threshold (float, optional):
                Hierarchy horizontal cut threshold. Defaults to None, returning a single labeling (no cut).
            min_area (int, optional): Minimum number of pixels per segment. Defaults to 50.
            max_area (int, optional): Maximum number of pixels per segment. Defaults to 1000000.
            min_frontier (float, optional): Minimum required edge weight for splitting segments. Defaults to 0.25.
            anisotropy_pen (float, optional): Edge weight penalization for the z-axis. Defaults to 0.0.
            cache (bool, optional): Enables cache of data structures. Defaults to False.
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

        tree, _, cut = self.data()
        cut = cut.horizontal_cut_from_altitude(self.cut_threshold)
        relabeled, _, _ = segmentation.relabel_sequential(
            cut.labelisation_leaves(tree), offset=offset
        )
        buffer[self.props.slice][self.props.image] = relabeled
        return len(cut.nodes())

    @property
    def cache(self) -> bool:
        return self._cache_active

    @cache.setter
    def cache(self, status: bool) -> None:
        self._cache_active = status
        self.props._cache_active = status
        if not status:
            self._cache.clear()
            self.props._cache.clear()
            self._nodes.clear()

    @staticmethod
    def _filter_contour_strength(
        tree: hg.Tree,
        alt: ArrayLike,
        graph: hg.UndirectedGraph,
        weights: ArrayLike,
        threshold: float,
    ) -> Tuple[hg.Tree, ArrayLike]:

        hg.set_attribute(graph, "no_border_vertex_out_degree", None)
        irrelevant_nodes = hg.attribute_contour_strength(tree, weights) < threshold
        hg.set_attribute(graph, "no_border_vertex_out_degree", 6)

        tree, node_map = hg.simplify_tree(tree, irrelevant_nodes)
        return tree, alt[node_map]

    def data(self) -> Tuple[hg.Tree, ArrayLike, hg.HorizontalCutExplorer]:
        """
        Creates and filters the watershed hierarchy.

        Raises:
            RuntimeError: It returns a error to avoid SegFault from Higra
            when processing with a single node. Increase `min_area` to avoid this.

        Returns:
            Tuple[hg.Tree, ArrayLike, hg.HorizontalCutExplorer]:
            It returns the watershed hierarchy (tree), its altitudes and multiple
            possible horizontal cuts.
        """
        if "tree" in self._cache and "alt" in self._cache and "cut" in self._cache:
            return self._cache["tree"], self._cache["alt"], self._cache["cut"]

        image = self.props.intensity_image
        if image.dtype == np.float16:
            image = image.astype(np.float32)

        if image.size < 8:
            raise RuntimeError(f"Region too small. Size of {image.size} found.")

        mask = self.props.image
        graph, weights = mask_to_graph(mask, image, self._anisotropy_pen)

        tree, alt = self.hierarchy_fun(graph, weights)
        tree, alt = hg.filter_small_nodes_from_tree(tree, alt, self._min_area)
        if self._min_frontier > 0.0:
            tree, alt = self._filter_contour_strength(
                tree, alt, graph, weights, self._min_frontier
            )

        cut = hg.HorizontalCutExplorer(tree, alt)
        if self._cache_active:
            self._cache["tree"] = tree
            self._cache["alt"] = alt
            self._cache["cut"] = cut

        return tree, alt, cut

    @property
    def tree(self) -> hg.Tree:
        tree = self._cache.get("tree")
        if tree is None:
            tree, _, _ = self.data()
        return tree

    @property
    def alt(self) -> ArrayLike:
        alt = self._cache.get("alt")
        if alt is None:
            _, alt, _ = self.data()
        return alt

    @property
    def cut(self) -> hg.HorizontalCutExplorer:
        cut = self._cache.get("cut")
        if cut is None:
            _, _, cut = self.data()
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
        area = hg.attribute_area(tree)
        num_leaves = tree.num_leaves()

        for i, node_idx in enumerate(
            tree.leaves_to_root_iterator(include_leaves=False)
        ):
            if area[num_leaves + i] > self._max_area:
                continue
            self._nodes[node_idx] = self.create_node(
                node_idx, self, area=area[num_leaves + i].item()
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
            root_index, self, area=self.props.area
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
    labels: ArrayLike, boundaries: ArrayLike, max_area: int
) -> ArrayLike:
    """
    This function oversegment segments given an maximum area to decrease the overall hierarchy volume (area),
    speeding up the remaining computation.


    Args:
        labels (ArrayLike): Input labels to be split into smaller segments.
        boundaries (ArrayLike): Weight graph.
        max_area (int): Maximum area (hierarchy threshold cut).

    Returns:
        ArrayLike: Labeled image with a greater number of labels than the input.
    """

    dtype = np.promote_types(np.float32, boundaries.dtype)
    boundaries = boundaries.astype(dtype)  # required by numba
    offset = 1
    new_labels = np.zeros_like(labels)
    for c in measure.regionprops(labels, boundaries):
        if c.area > max_area:
            graph, weights = mask_to_graph(c.image, c.intensity_image, 0.0)
            tree, alt = hg.watershed_hierarchy_by_area(graph, weights)
            cut = hg.labelisation_horizontal_cut_from_threshold(tree, alt, max_area)
            cut, _, _ = segmentation.relabel_sequential(cut, offset=offset)
            offset = cut.max() + 1
            new_labels[c.slice][c.image] = cut
        else:
            new_labels[c.slice][c.image] = offset
            offset += 1
    return new_labels


def create_hierarchies(
    labels: ArrayLike,
    boundaries: ArrayLike,
    cache: bool = False,
    **kwargs,
) -> List[Hierarchy]:
    assert issubclass(labels.dtype.type, np.integer) or labels.dtype == bool

    labels = measure.label(labels, connectivity=1)
    if "min_area" in kwargs:
        morphology.remove_small_objects(
            labels, min_size=kwargs["min_area"], in_place=True
        )

    if "max_area" in kwargs:
        labels = oversegment_components(labels, boundaries, kwargs["max_area"])

    return [
        Hierarchy(c, cache=cache, **kwargs)
        for c in measure.regionprops(labels, boundaries, cache=cache)
    ]


def to_labels(
    hierarchies: List[Hierarchy], shape: Tuple[int], dtype: np.dtype = np.int32
) -> None:
    labels = np.zeros(shape, dtype=dtype)

    offset = 1
    for h in hierarchies:
        offset += h.label_buffer(labels, offset=offset)

    return labels
