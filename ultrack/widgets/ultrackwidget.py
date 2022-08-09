from pathlib import Path

import napari
from magicgui.widgets import Container

from ultrack.config.config import MainConfig, load_config
from ultrack.widgets.computewidget import ComputeWidget
from ultrack.widgets.initwidget import InitWidget
from ultrack.widgets.mainconfigwidget import MainConfigWidget


class UltrackWidget(Container):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__(labels=False)

        self._viewer = viewer

        config = MainConfig()

        self._main_config_w = MainConfigWidget(config=config)
        self._main_config_w._config_loader_w.changed.connect(self._on_config_loaded)
        self.append(self._main_config_w)

        self._init_w = InitWidget(config=config.init_config)
        self.append(self._init_w)

        self._compute_w = ComputeWidget(config=config.compute_config)
        self.append(self._compute_w)

    @property
    def config(self) -> MainConfig:
        return self._main_config_w.config

    @config.setter
    def config(self, value: MainConfig) -> None:
        self._main_config_w.config = value
        self._init_w.config = value.init_config
        self._compute_w.config = value.compute_config

    def _on_config_loaded(self, value: Path) -> None:
        self.config = load_config(value)


if __name__ == "__main__":
    viewer = napari.Viewer()
    widget = UltrackWidget(viewer)
    viewer.window.add_dock_widget(widget)
    napari.run()
