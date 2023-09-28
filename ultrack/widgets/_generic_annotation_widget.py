import enum
import logging
from abc import abstractmethod
from typing import List, Optional

import napari
from magicgui.widgets import ComboBox, PushButton
from napari.layers import Image, Labels

from ultrack.core.database import NodeDB, is_table_empty
from ultrack.core.segmentation.node import Node
from ultrack.widgets._generic_data_widget import GenericDataWidget

logging.basicConfig()
logging.getLogger("sqlachemy.engine").setLevel(logging.INFO)

LOG = logging.getLogger(__name__)


class GenericAnnotationWidget(GenericDataWidget):
    def __init__(
        self,
        viewer: napari.Viewer,
        label: str,
        annot_opts: enum.IntEnum,
        suffix: str,
        sample_size: int = 50,
    ) -> None:

        # before init due to config initialization
        self._nodes: List[Node] = []
        self._sample_size = sample_size
        self._suffix = suffix
        self._mask_layer_name = f"Current Node Mask ~ {self._suffix}"

        self._next_btn = PushButton(text="Next")
        self._prev_btn = PushButton(text="Prev")
        self._confirm_btn = PushButton(text="Confirm", enabled=False)

        super().__init__(viewer, label)

        self._next_btn.changed.connect(self._on_next)
        self.append(self._next_btn)

        self._prev_btn.changed.connect(self._on_prev)
        self.append(self._prev_btn)

        self._annot_w = ComboBox(label="Annot.", choices=annot_opts)
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

    @abstractmethod
    def _query_samples(self) -> List[Node]:
        pass

    def _is_valid_index_at_list_end(self, index: int) -> bool:
        if index >= len(self._nodes) and not is_table_empty(self.config, NodeDB):
            self._nodes += self._query_samples()
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
                    state["visible"] = self._viewer.layers[crop_name].visible
                    self._viewer.layers.remove(crop_name)

                data = node.roi(data[node.time])

                state["translate"] = (
                    state["translate"][-data.ndim :] + node.bbox[: data.ndim]
                )
                state["scale"] = state["scale"][-data.ndim :]
                state["name"] = crop_name

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
                node.mask,
                name=self._mask_layer_name,
                translate=node.bbox[: node.mask.ndim],
            )
            layer.bind_key("Enter", self._on_confirm)

        self._annot_w.value = self.get_annotation(node.id)
        self._viewer.layers.move(self._viewer.layers.index(self._mask_layer_name), -1)

        if self._viewer.dims.ndim == node.mask.ndim:
            self._viewer.dims.set_point(range(len(node.centroid)), node.centroid)
        else:
            self._viewer.dims.set_point(
                range(len(node.centroid) + 1), (node.time, *node.centroid)
            )

        self._viewer.camera.center = node.centroid

    def _on_confirm(self, layer: Optional[Labels] = None) -> None:
        if not self._confirm_btn.enabled:
            # required by key binding
            return

        # layer required by napari bind_key
        self.set_annotation(self._nodes[self.list_index].id, self._annot_w.value)

        if self._next_btn.enabled:
            self._next_btn.clicked.emit()

    @abstractmethod
    def get_annotation(self, index: int) -> enum.IntEnum:
        pass

    @abstractmethod
    def set_annotation(self, index: int, annot: enum.IntEnum) -> None:
        pass
