from magicgui.widgets import FloatSpinBox, PushButton, SpinBox

from ultrack.config.config import LinkingConfig
from ultrack.widgets.baseconfigwidget import BaseConfigWidget


class LinkingWidget(BaseConfigWidget):
    def __init__(self, config: LinkingConfig):
        super().__init__(label="Linking", config=config)

    def _setup_widgets(self) -> None:
        self._attr_to_widget = {
            "max_neighbors": SpinBox(label="Max. neighbors"),
            "max_distance": FloatSpinBox(label="Max. distance"),
            "n_workers": SpinBox(label="Num. workers"),
        }
        for widget in self._attr_to_widget.values():
            self.append(widget)

        self._link_btn = PushButton(text="Link")
        self.append(self._link_btn)
