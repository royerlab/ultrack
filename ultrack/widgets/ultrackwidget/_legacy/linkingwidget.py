from magicgui.widgets import CheckBox, FloatSpinBox, PushButton, SpinBox

from ultrack.config.config import LinkingConfig
from ultrack.widgets.ultrackwidget._legacy.baseconfigwidget import BaseConfigWidget


class LinkingWidget(BaseConfigWidget):
    def __init__(self, config: LinkingConfig):
        super().__init__(label="Linking", config=config)
        self._images_w = CheckBox(
            label="Use selected layers",
            value=False,
            tooltip="When checked, the selected layers are used to compute edge weights with "
            "discrete cosine transform, otherwise the IoU of the segments are used.",
        )
        self.append(self._images_w)

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
