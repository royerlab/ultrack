from pathlib import Path
from typing import List, Sequence
from warnings import warn

import napari
import numpy as np
import sqlalchemy as sqla
from magicgui.widgets import FileEdit, FloatSlider, PushButton
from sqlalchemy.orm import Session

from ultrack.config import DataConfig, load_config
from ultrack.core.database import NodeDB
from ultrack.core.segmentation.node import Node
from ultrack.widgets.ultrackwidget.baseconfigwidget import BaseConfigWidget


class SegmVizWidget(BaseConfigWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__(DataConfig(), "Segm. Viz.")
        self._viewer = viewer
        self._layer_name = "Segm. Hypothesis"
        self._nodes: List[Node] = []
        self._hier_ids: List[int] = []

        self._config_loader_w = FileEdit(
            filter="*toml",
            label="Config. Path",
            value=None,
        )
        self.append(self._config_loader_w)

        self._load_btn = PushButton(text="Load Segm.")
        self._load_btn.changed.connect(self._on_load_segm)
        self.append(self._load_btn)

        self._area_threshold_w = FloatSlider(label="Area threshold", min=0, max=0)
        self._area_threshold_w.changed.connect(self._on_threshold_update)
        self.append(self._area_threshold_w)

    def _setup_widgets(self) -> None:
        pass

    def _on_config_loaded(self, value: Path) -> None:
        if value.exists() and value.is_file():
            self.config = load_config(value).data_config

    @BaseConfigWidget.config.setter
    def config(self, value: DataConfig) -> None:
        BaseConfigWidget.config.fset(self, value)
        self._ndim = len(self._shape)

    @property
    def _shape(self) -> Sequence[int]:
        return self.config.metadata.get("shape", [])

    def _on_load_segm(self) -> None:
        time = self._time
        engine = sqla.create_engine(self.config.database_path)
        with Session(engine) as session:
            query = (
                session.query(NodeDB.pickle, NodeDB.t_hier_id)
                .where(NodeDB.t == time)
                .order_by(NodeDB.area)
            )
            self._nodes, self._hier_ids = zip(*query)
            # overlaps = (
            #     session.query(OverlapDB)
            #     .join(NodeDB, NodeDB.id == OverlapDB.node_id)
            #     .where(NodeDB.t == time)
            # )

        if len(self._nodes) == 0:
            raise ValueError(f"Could not find segmentations at time {time}")

        area = np.asarray([node.area for node in self._nodes])
        self._area_threshold_w.min = area.min()
        self._area_threshold_w.max = area.max()
        self._area_threshold_w.value = np.median(area)

    def _on_threshold_update(self, value: float) -> None:
        segmentation = self._get_segmentation(threshold=value)
        if self._layer_name in self._viewer.layers:
            self._viewer.layers[self._layer_name].data = segmentation
        else:
            self._viewer.add_labels(segmentation, name=self._layer_name)

    def _get_segmentation(self, threshold: float) -> np.ndarray:
        """
        NOTE:
        when making this interactive it could be interesting to use the overlap data
        to avoid empty regions when visualizing segments
        """
        if self._ndim == 0:
            raise ValueError(
                "Could not find `shape` metadata. It should be saved during `segmentation` on your `workdir`."
            )

        seen_hierarchies = set()

        buffer = np.zeros(self._shape[1:], dtype=np.uint32)  # ignoring time
        for node, hier_id in zip(self._nodes, self._hier_ids):
            if node.area <= threshold or hier_id not in seen_hierarchies:
                # paint segments larger than threshold on empty regions
                node.paint_buffer(buffer, node.id, include_time=False)
                seen_hierarchies.add(hier_id)

        return buffer

    @property
    def _time(self) -> None:
        available_ndim = self._viewer.dims.ndim
        if available_ndim < self._ndim:
            warn(
                "Napari `ndims` smaller than dataset `ndims`. "
                f"Expected {self._ndim}, found {available_ndim}. Using time = 0"
            )
            return 0

        return self._viewer.dims.point[-self._ndim]
