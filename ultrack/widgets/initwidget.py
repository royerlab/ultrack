from magicgui.widgets import ComboBox, FloatSpinBox, PushButton, SpinBox

from ultrack.config.config import NAME_TO_WS_HIER, InitConfig
from ultrack.widgets.baseconfigwidget import BaseConfigWidget


class InitWidget(BaseConfigWidget):
    def __init__(self, config: InitConfig):
        super().__init__(label="Initialization param.", config=config)

    def _setup_widgets(self) -> None:
        self._attr_to_widget = {
            "threshold": FloatSpinBox(label="Det. threshold"),
            "min_area": SpinBox(label="Min. area"),
            "max_area": SpinBox(label="Max. area", max=1_000_000_000),
            "min_frontier": FloatSpinBox(label="Min. frontier"),
            "anisotropy_penalization": FloatSpinBox(
                label="Anisotropy pen.", min=-100, max=100
            ),
            "ws_hierarchy": ComboBox(
                label="Watershed by", choices=list(NAME_TO_WS_HIER.items())
            ),
            "n_workers": SpinBox(label="Num. workers"),
            "max_neighbors": SpinBox(label="Max. neighbors"),
            "max_distance": FloatSpinBox(label="Max. distance"),
        }
        for widget in self._attr_to_widget.values():
            self.append(widget)

        self._build_w = PushButton(text="Build")
        self.append(self._build_w)

        self._load_w = PushButton(text="Load")
        self.append(self._load_w)
