from magicgui.widgets import FileEdit

from ultrackBeehive.widgets.ultrackwidget._legacy.baseconfigwidget import BaseConfigWidget


class DataWidget(BaseConfigWidget):
    def _setup_widgets(self) -> None:
        self._attr_to_widget = {
            "working_dir": FileEdit(mode="d", label="Working dir."),
        }

        for widget in self._attr_to_widget.values():
            self.append(widget)
