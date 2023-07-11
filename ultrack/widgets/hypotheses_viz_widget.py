import logging
from typing import Dict, List, Optional, Sequence
from warnings import warn

import napari
import numpy as np
import pandas as pd
import sqlalchemy as sqla
from magicgui.widgets import CheckBox, FloatSlider, PushButton
from napari.layers import Labels
from sqlalchemy.orm import Session

from ultrack.core.database import LinkDB, NodeDB
from ultrack.core.segmentation.node import Node
from ultrack.widgets._generic_data_widget import GenericDataWidget

logging.basicConfig()
logging.getLogger("sqlachemy.engine").setLevel(logging.INFO)

LOG = logging.getLogger(__name__)


class HypothesesVizWidget(GenericDataWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__(viewer, "Hypotheses Viz.")
        self._segm_layer_name = "Segm. Hypotheses"
        self._link_layer_name = "Links"
        self._nodes: Dict[int, Node] = {}
        self._hier_ids: List[int] = []

        self._load_btn = PushButton(text="Load Segm.")
        self._load_btn.changed.connect(self._on_load_segm)
        self.append(self._load_btn)

        self._area_threshold_w = FloatSlider(label="Area threshold", min=0, max=0)
        self._area_threshold_w.changed.connect(self._on_threshold_update)
        self.append(self._area_threshold_w)

        self._link_w = CheckBox(text="Show links", value=False)
        self.append(self._link_w)

    def _on_config_changed(self) -> None:
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

        self._nodes = {node.id: node for node in self._nodes}

        if len(self._nodes) == 0:
            raise ValueError(f"Could not find segmentations at time {time}")

        area = np.asarray([node.area for node in self._nodes.values()])
        self._area_threshold_w.min = area.min()
        self._area_threshold_w.max = area.max()
        self._area_threshold_w.value = np.median(area)

    def _on_threshold_update(self, value: float) -> None:
        segmentation = self._get_segmentation(threshold=value)
        if self._segm_layer_name in self._viewer.layers:
            self._viewer.layers[self._segm_layer_name].data = segmentation
        else:
            layer = self._viewer.add_labels(segmentation, name=self._segm_layer_name)
            layer.mouse_move_callbacks.append(self._on_mouse_move)

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
        for node, hier_id in zip(self._nodes.values(), self._hier_ids):
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

    def _on_mouse_move(self, layer: Optional[Labels], event) -> None:
        if not self._link_w.value:
            return
        self._load_neighbors(layer.get_value(event.position, world=True))

    def _load_neighbors(self, index: int) -> None:
        if index is None or index <= 0:
            return

        index = int(index)  # might be numpy array

        LOG.info(f"Loading node index = {index}")

        engine = sqla.create_engine(self.config.database_path, echo=True)
        with Session(engine) as session:
            query = session.query(NodeDB.z, NodeDB.y, NodeDB.x, LinkDB.weight).where(
                LinkDB.target_id == NodeDB.id, LinkDB.source_id == index
            )
            df = pd.read_sql(query.statement, session.bind)

        LOG.info(f"Found {len(df)} neighbors")

        if len(df) == 0:
            return

        node = self._nodes[index]
        ndim = len(node.centroid)
        centroids = df[["z", "y", "x"]].values[:, -ndim:]  # removing z if 2D

        vectors = np.tile(self._nodes[index].centroid, (len(df), 2, 1))
        vectors[:, 1, :] = centroids - vectors[:, 0, :]

        if self._link_layer_name in self._viewer.layers:
            self._viewer.layers.remove(self._link_layer_name)

        self._viewer.add_vectors(
            data=vectors,
            name=self._link_layer_name,
            features={"weights": df["weight"]},
            edge_color="weights",
            opacity=1.0,
        )

        LOG.info(f"vectors:\n{vectors}")

        self._viewer.layers.selection.active = self._viewer.layers[
            self._segm_layer_name
        ]
