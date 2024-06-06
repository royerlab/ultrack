from magicgui.widgets import ComboBox, FloatSpinBox, PushButton, SpinBox

from ultrack.config.segmentationconfig import NAME_TO_WS_HIER, SegmentationConfig
from ultrack.widgets.ultrackwidget._legacy.baseconfigwidget import BaseConfigWidget


class SegmentationWidget(BaseConfigWidget):
    def __init__(self, config: SegmentationConfig):
        super().__init__(label="Segmentation", config=config)

    def _setup_widgets(self) -> None:
        self._attr_to_widget = {
            "threshold": FloatSpinBox(label="Det. threshold"),
            "min_area": SpinBox(label="Min. area", max=1_000_000_000),
            "max_area": SpinBox(label="Max. area", max=1_000_000_000),
            "min_frontier": FloatSpinBox(label="Min. frontier"),
            "anisotropy_penalization": FloatSpinBox(
                label="Anisotropy pen.", min=-100, max=100
            ),
            "ws_hierarchy": ComboBox(
                label="Watershed by", choices=list(NAME_TO_WS_HIER.items())
            ),
            "n_workers": SpinBox(label="Num. workers"),
        }
        for widget in self._attr_to_widget.values():
            self.append(widget)

        self._segment_btn = PushButton(text="Segment")
        self.append(self._segment_btn)
