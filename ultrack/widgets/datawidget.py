from magicgui.widgets import LineEdit

from ultrack.widgets.baseconfigwidget import BaseConfigWidget


class DataWidget(BaseConfigWidget):
    def _setup_widgets(self) -> None:
        self._attr_to_widget = {
            "working_dir": LineEdit(label="Working dir."),
        }

        for widget in self._attr_to_widget.values():
            self.append(widget)
