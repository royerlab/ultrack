from magicgui.widgets import FileEdit, LineEdit, create_widget
from napari.layers import Image

from ultrack.config import MainConfig
from ultrack.widgets.baseconfigwidget import BaseConfigWidget


class MainConfigWidget(BaseConfigWidget):
    def __init__(self, config: MainConfig):
        super().__init__(label="ULTRACK", config=config)

    def _setup_widgets(self) -> None:
        self._detection_layer_w = create_widget(annotation=Image, label="Detection")
        self.append(self._detection_layer_w)

        self._edge_layer_w = create_widget(annotation=Image, label="Edge")
        self.append(self._edge_layer_w)

        self._config_loader_w = FileEdit(filter=".toml", label="Load config.")
        self.append(self._config_loader_w)

        self._attr_to_widget = {
            "working_dir": LineEdit(label="Working dir."),
        }

        for widget in self._attr_to_widget.values():
            self.append(widget)
