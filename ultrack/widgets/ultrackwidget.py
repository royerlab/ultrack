from pathlib import Path

import napari
from magicgui.widgets import Container

from ultrack.config.config import MainConfig, load_config
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

        self._segmentation_w = SegmentationWidget(config=config.segmentation_config)
        self.append(self._segmentation_w)

        self._linking_w = LinkingWidget(config=config.linking_config)
        self.append(self._linking_w)

        self._tracking_w = TrackingWidget(config=config.tracking_config)
        self.append(self._tracking_w)

    @property
    def config(self) -> MainConfig:
        return self._main_config_w.config

    @config.setter
    def config(self, value: MainConfig) -> None:
        self._main_config_w.config = value
        self._segmentation_w.config = value.segmentation_config
        self._linking_w.config = value.linking_config
        self._tracking_w.config = value.tracking_config

    def _on_config_loaded(self, value: Path) -> None:
        self.config = load_config(value)


if __name__ == "__main__":
    viewer = napari.Viewer()
    widget = UltrackWidget(viewer)
    viewer.window.add_dock_widget(widget)
    napari.run()
