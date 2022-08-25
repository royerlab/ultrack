from pathlib import Path

import napari
from magicgui.widgets import Container

from ultrack import link, segment, track
from ultrack.config.config import MainConfig, load_config
from ultrack.core.export.tracks_layer import to_tracks_layer
from ultrack.widgets.datawidget import DataWidget
from ultrack.widgets.linkingwidget import LinkingWidget
from ultrack.widgets.mainconfigwidget import MainConfigWidget
from ultrack.widgets.segmentationwidget import SegmentationWidget
from ultrack.widgets.trackingwidget import TrackingWidget


class UltrackWidget(Container):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__(labels=False)

        self._viewer = viewer

        config = MainConfig()

        self._main_config_w = MainConfigWidget(config=config)
        self._main_config_w._config_loader_w.changed.connect(self._on_config_loaded)
        self.append(self._main_config_w)

        self._data_config_w = DataWidget(config=config.data_config)
        self.append(self._data_config_w)

        self._segmentation_w = SegmentationWidget(config=config.segmentation_config)
        self.append(self._segmentation_w)

        self._linking_w = LinkingWidget(config=config.linking_config)
        self.append(self._linking_w)

        self._tracking_w = TrackingWidget(config=config.tracking_config)
        self.append(self._tracking_w)

        self._setup_signals()

    def _setup_signals(self) -> None:
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
        self.config = load_config(value)

    def _on_segment(self) -> None:
        segment(
            detection=self._main_config_w._detection_layer_w.value.data,
            edge=self._main_config_w._edge_layer_w.value.data,
            segmentation_config=self._segmentation_w.config,
            data_config=self._data_config_w.config,
        )

    def _on_link(self) -> None:
        link(self._linking_w.config, self._data_config_w.config)

    def _on_track(self) -> None:
        track(self._tracking_w.config, self._data_config_w.config)
        tracks, graph = to_tracks_layer(self._data_config_w.config)
        self._viewer.add_tracks(tracks, graph=graph)
