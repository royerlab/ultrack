import logging
from typing import List, Optional

import napari
from magicgui.widgets import ComboBox, PushButton
from napari.layers import Image, Labels
from sqlalchemy import create_engine, func
from sqlalchemy.orm import Session

from ultrack.core.database import (
    NodeAnnotation,
    NodeDB,
    get_node_annotation,
    is_table_empty,
    set_node_annotation,
)
from ultrack.core.segmentation.node import Node
from ultrack.widgets._generic_data_widget import GenericDataWidget

logging.basicConfig()
logging.getLogger("sqlachemy.engine").setLevel(logging.INFO)

LOG = logging.getLogger(__name__)


class NodeAnnotationWidget(GenericDataWidget):
    def __init__(self, viewer: napari.Viewer) -> None:

        # before init due to config initialization
        self._nodes: List[Node] = []
        self._sample_size = 50
        self._suffix = " ~ ultrack"
        self._mask_layer_name = "Current Node Mask ~ ultrack"

        self._next_btn = PushButton(text="Next")
        self._prev_btn = PushButton(text="Prev")
        self._confirm_btn = PushButton(text="Confirm", enabled=False)

        super().__init__(viewer, "Node Annotation")

        self._next_btn.changed.connect(self._on_next)
        self.append(self._next_btn)

        self._prev_btn.changed.connect(self._on_prev)
        self.append(self._prev_btn)

        self._annot_w = ComboBox(label="Annot.", choices=NodeAnnotation)
        self.append(self._annot_w)

        self._confirm_btn.changed.connect(self._on_confirm)
        self.append(self._confirm_btn)

        self.list_index = 0

    def _on_config_changed(self) -> None:
        self._nodes: List[Node] = []

        for layer in list(self._viewer.layers):
            if layer.name.endswith(self._suffix):
                self._viewer.layers.remove(layer.name)

        # must be after cleanup
        self.list_index = 0

    def _is_valid_index_at_list_end(self, index: int) -> bool:
        if index >= len(self._nodes) and not is_table_empty(self.config, NodeDB):
            engine = create_engine(self.config.database_path)
            with Session(engine) as session:
                nodes = (
                    session.query(NodeDB.pickle)
                    .where(NodeDB.id.not_in([node.id for node in self._nodes]))
                    .order_by(func.random())
                    .limit(self._sample_size)
                    .all()
                )
            self._nodes += [node for node, in nodes]

        return index < len(self._nodes) - 1

    @property
    def list_index(self) -> int:
        return self._list_index

    @list_index.setter
    def list_index(self, value: int) -> int:
        self._list_index = value
        self._prev_btn.enabled = value > 0
        self._next_btn.enabled = self._is_valid_index_at_list_end(value)
        self._confirm_btn.enabled = len(self._nodes) > 0
        if 0 <= self._list_index < len(self._nodes):
            self._load_node(self._nodes[self._list_index])

    def _on_next(self) -> None:
        self.list_index += 1

    def _on_prev(self) -> None:
        self.list_index -= 1

    def _load_node(self, node: Node) -> None:

        shape = tuple(self.config.metadata["shape"])
        for layer in list(self._viewer.layers):
            if (
                isinstance(layer, (Image, Labels))
                and layer.data.shape == shape
                and not layer.name.endswith(self._suffix)
            ):

                data, state, type_str = layer.as_layer_data_tuple()
                name = state["name"]
                crop_name = name + self._suffix
                if crop_name in self._viewer.layers:
                    self._viewer.layers.remove(crop_name)

                data = node.roi(data[node.time])

                state["translate"] = (
                    state["translate"][-data.ndim :] + node.bbox[: data.ndim]
                )
                state["scale"] = state["scale"][-data.ndim :]
                state["name"] = crop_name
                state["visible"] = True

                # removing this matrices for now
                state.pop("rotate")
                state.pop("shear")
                state.pop("affine")

                # was causing issues between 2d and 3d
                state.pop("interpolation", None)

                self._viewer._add_layer_from_data(data, state, type_str)

        try:
            layer = self._viewer.layers[self._mask_layer_name]
            layer.data = node.mask
            layer.translate = node.bbox[: node.mask.ndim]
        except KeyError:
            layer = self._viewer.add_labels(
                node.mask, name=self._mask_layer_name, translate=node.bbox[:3]
            )
            layer.bind_key("Enter", self._on_confirm)

        self._annot_w.value = get_node_annotation(self.config, node.id)
        self._viewer.layers.move(self._viewer.layers.index(self._mask_layer_name), -1)
        self._viewer.dims.set_point(range(4), (node.time, *node.centroid))

    def _on_confirm(self, layer: Optional[Labels] = None) -> None:
        if not self._confirm_btn.enabled:
            # required by key binding
            return

        # layer required by napari bind_key
        set_node_annotation(
            self.config, self._nodes[self.list_index].id, self._annot_w.value.name
        )
        if self._next_btn.enabled:
            self._next_btn.clicked.emit()
