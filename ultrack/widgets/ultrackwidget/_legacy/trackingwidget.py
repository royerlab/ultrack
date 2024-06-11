from magicgui.widgets import FloatSpinBox, PushButton

from ultrack.config.config import TrackingConfig
from ultrack.widgets.ultrackwidget._legacy.baseconfigwidget import BaseConfigWidget


class TrackingWidget(BaseConfigWidget):
    def __init__(self, config: TrackingConfig):
        super().__init__(label="Tracking", config=config)

    def _setup_widgets(self) -> None:
        self._attr_to_widget = {
            "appear_weight": FloatSpinBox(label="Appear weight", min=-100, max=100),
            "disappear_weight": FloatSpinBox(
                label="Disappear weight", min=-100, max=100
            ),
            "division_weight": FloatSpinBox(label="Division weight", min=-100, max=100),
            "power": FloatSpinBox(label="Power", min=0.1, max=10),
            "bias": FloatSpinBox(label="Bias", min=-100, max=100),
        }
        for widget in self._attr_to_widget.values():
            self.append(widget)
            if isinstance(widget, FloatSpinBox):
                # increasing number of decimals
                # reference: https://github.com/napari/magicgui/issues/226
                spinbox = widget._widget._qwidget
                spinbox.setDecimals(4)

        self._track_btn = PushButton(text="Track")
        self.append(self._track_btn)
