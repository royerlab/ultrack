from abc import abstractmethod
from pathlib import Path

import napari
from magicgui.widgets import FileEdit

from ultrack.config import DataConfig, load_config
from ultrack.widgets.ultrackwidget._legacy.baseconfigwidget import BaseConfigWidget


class GenericDataWidget(BaseConfigWidget):
    def __init__(self, viewer: napari.Viewer, label: str) -> None:
        self._viewer = viewer
        super().__init__(DataConfig(), label)

        self._config_loader_w = FileEdit(
            filter="*toml",
            label="Config. Path",
            value=None,
        )
        self.append(self._config_loader_w)

    def _setup_widgets(self) -> None:
        pass

    def _on_config_loaded(self, value: Path) -> None:
        if value.exists() and value.is_file():
            self.config = load_config(value).data_config

    @BaseConfigWidget.config.setter
    def config(self, value: DataConfig) -> None:
        BaseConfigWidget.config.fset(self, value)
        self._on_config_changed()

    @abstractmethod
    def _on_config_changed(self) -> None:
        pass
