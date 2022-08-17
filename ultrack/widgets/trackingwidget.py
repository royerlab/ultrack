from magicgui.widgets import FloatSpinBox, PushButton

from ultrack.config.config import TrackingConfig
from ultrack.widgets.baseconfigwidget import BaseConfigWidget


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
        }
        for widget in self._attr_to_widget.values():
            self.append(widget)

        self._run_w = PushButton(text="Track")
        self.append(self._run_w)
