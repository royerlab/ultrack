import logging
from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import napari
import numpy as np
import pandas as pd
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from numpy.typing import ArrayLike
from qtpy.QtWidgets import QFrame, QTabWidget, QVBoxLayout, QWidget

from ultrack import link, segment, solve
from ultrack.config.config import MainConfig, load_config
from ultrack.core.database import LinkDB, NodeDB, is_table_empty
from ultrack.core.export.tracks_layer import to_tracks_layer
from ultrack.core.export.zarr import tracks_to_zarr
from ultrack.widgets.ultrackwidget._legacy.datawidget import DataWidget
from ultrack.widgets.ultrackwidget._legacy.linkingwidget import LinkingWidget
from ultrack.widgets.ultrackwidget._legacy.mainconfigwidget import MainConfigWidget
from ultrack.widgets.ultrackwidget._legacy.segmentationwidget import SegmentationWidget
from ultrack.widgets.ultrackwidget._legacy.trackingwidget import TrackingWidget
from ultrack.widgets.utils import wait_cursor

LOG = logging.getLogger(__name__)


class UltrackWidget(QWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()

        layout = QVBoxLayout()

        self._viewer = viewer
        self._tab = QTabWidget()
        self.setLayout(layout)

        layout.addWidget(self._tab)

        config = MainConfig()

        self._main_config_w = MainConfigWidget(config=config)
        tmp_layout = self._main_config_w.native.layout()
        tmp_layout.takeAt(tmp_layout.count() - 1)
        self._data_config_w = DataWidget(config=config.data_config)

        frame_layout = QVBoxLayout()
        main_frame = QFrame()
        main_frame.setLayout(frame_layout)
        frame_layout.addWidget(self._main_config_w.native)
        frame_layout.addWidget(self._data_config_w.native)
        frame_layout.addStretch()
        self._tab.addTab(main_frame, "Main")

        self._segmentation_w = SegmentationWidget(config=config.segmentation_config)
        self._tab.addTab(self._segmentation_w.native, "Segmentation")

        self._linking_w = LinkingWidget(config=config.linking_config)
        self._tab.addTab(self._linking_w.native, "Linking")

        self._tracking_w = TrackingWidget(config=config.tracking_config)
        self._tab.addTab(self._tracking_w.native, "Tracking")

        self._setup_signals()
        self._update_widget_status()

    def _setup_signals(self) -> None:
        self._main_config_w._config_loader_w.changed.connect(self._on_config_loaded)
        self._main_config_w._config_loader_w.changed.connect(self._update_widget_status)
        self._main_config_w._detection_layer_w.changed.connect(
            self._update_widget_status
        )
        self._main_config_w._edge_layer_w.changed.connect(self._update_widget_status)
        self._segmentation_w._segment_btn.changed.connect(self._on_segment)
        self._linking_w._link_btn.changed.connect(self._on_link)
        self._tracking_w._track_btn.changed.connect(self._on_track)

    @property
    def config(self) -> MainConfig:
        return self._main_config_w.config

    @config.setter
    def config(self, value: MainConfig) -> None:
        self._main_config_w.config = value
        self._data_config_w.config = value.data_config
        self._segmentation_w.config = value.segmentation_config
        self._linking_w.config = value.linking_config
        self._tracking_w.config = value.tracking_config

    def _on_config_loaded(self, value: Path) -> None:
        if value.exists() and value.is_file():
            self.config = load_config(value)

    def _on_segment(self) -> None:
        segmentation_worker = self._make_segmentation_worker()
        segmentation_worker.started.connect(self._lock_buttons)
        segmentation_worker.finished.connect(self._update_widget_status)
        segmentation_worker.start()

    def _on_link(self) -> None:
        if self._linking_w._images_w.value:
            for layer in self._viewer.layers.selection:
                if not isinstance(layer, (Image, Labels)):
                    raise ValueError(
                        f"Selected layers must be Image or Labels, {layer} not valid."
                    )
            images = tuple(layer.data for layer in self._viewer.layers.selection)
        else:
            images = tuple()

        LOG.info(f"Using {images} for linking")

        link_worker = self._make_link_worker(images=images)
        link_worker.started.connect(self._lock_buttons)
        link_worker.finished.connect(self._update_widget_status)
        link_worker.start()

    def _on_track(self) -> None:
        track_worker = self._make_track_worker()
        track_worker.started.connect(self._lock_buttons)
        track_worker.returned.connect(self._add_tracking_result)
        track_worker.finished.connect(self._update_widget_status)
        track_worker.start()

    @thread_worker
    @wait_cursor()
    def _make_segmentation_worker(self) -> None:
        segment(
            detection=self._main_config_w._detection_layer_w.value.data,
            edge=self._main_config_w._edge_layer_w.value.data,
            config=self.config,
            overwrite=True,
        )

    @thread_worker
    @wait_cursor()
    def _make_link_worker(self, images: Sequence[ArrayLike]) -> None:
        link(
            self.config,
            images=images,
            overwrite=True,
        )

    @thread_worker
    @wait_cursor()
    def _make_track_worker(self) -> Tuple[pd.DataFrame, Dict, np.ndarray]:
        solve(self.config, overwrite=True)
        tracks, graph = to_tracks_layer(self.config)
        labels = tracks_to_zarr(self.config)
        return tracks, graph, labels

    @wait_cursor()
    def _add_tracking_result(
        self, result: Tuple[pd.DataFrame, Dict, np.ndarray]
    ) -> None:
        tracks, graph, labels = result
        self._viewer.add_tracks(tracks, graph=graph)
        self._viewer.add_labels(labels)

    def _lock_buttons(self) -> None:
        LOG.info("Locking buttons")
        self._linking_w._link_btn.enabled = False
        self._tracking_w._track_btn.enabled = False
        self._segmentation_w._segment_btn.enabled = False

    def _update_widget_status(self) -> None:
        LOG.info("Update widget status")
        self._segmentation_w._segment_btn.enabled = (
            self._main_config_w._detection_layer_w.value is not None
            and self._main_config_w._edge_layer_w.value is not None
        )
        data_config = self._data_config_w.config
        self._linking_w._link_btn.enabled = not is_table_empty(data_config, NodeDB)
        self._tracking_w._track_btn.enabled = not is_table_empty(data_config, LinkDB)

    def reset_choices(self, *_: Any) -> None:
        self._main_config_w.reset_choices()
        self._data_config_w.reset_choices()
        self._linking_w.reset_choices()
        self._tracking_w.reset_choices()
        self._segmentation_w.reset_choices()
