from magicgui.widgets import FloatSpinBox, PushButton

from ultrack.config.config import ComputeConfig
from ultrack.widgets.baseconfigwidget import BaseConfigWidget


class ComputeWidget(BaseConfigWidget):
    def __init__(self, config: ComputeConfig):
        super().__init__(label="Compute param.", config=config)

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

        self._run_w = PushButton(text="Run")
        self.append(self._run_w)
