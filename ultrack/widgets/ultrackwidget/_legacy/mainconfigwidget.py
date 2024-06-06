import toml
from magicgui.widgets import FileEdit, PushButton, create_widget
from napari.layers import Image

from ultrack.config import MainConfig
from ultrack.widgets.ultrackwidget._legacy.baseconfigwidget import BaseConfigWidget


class MainConfigWidget(BaseConfigWidget):
    def __init__(self, config: MainConfig):
        super().__init__(label="ULTRACK", config=config)

    def _setup_widgets(self) -> None:
        self._foreground_layer_w = create_widget(annotation=Image, label="Foreground")
        self.append(self._foreground_layer_w)

        self._edge_layer_w = create_widget(annotation=Image, label="Edge")
        self.append(self._edge_layer_w)

        self._config_loader_w = FileEdit(
            filter="*.toml", label="Config. path", value="config.toml"
        )
        self.append(self._config_loader_w)

        self._save_config_btn = PushButton(text="Save config")
        self._save_config_btn.changed.connect(self._on_save_config)
        self.append(self._save_config_btn)

    def _on_save_config(self) -> None:
        with open(self._config_loader_w.value, mode="w") as f:
            toml.dump(self.config.dict(by_alias=True), f)
