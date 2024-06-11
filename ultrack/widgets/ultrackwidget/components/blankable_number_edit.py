from qtpy.QtWidgets import QLineEdit

from ultrack.widgets.ultrackwidget.components.number_validators import (
    DoubleOrEmptyValidator,
    IntOrEmptyValidator,
)


class BlankableNumberEdit(QLineEdit):
    def __init__(
        self, parent=None, default=0, minimum=-999999, maximum=999999, dtype=int
    ):
        super().__init__(parent)

        self.dtype = dtype
        if dtype == int:
            self.setValidator(IntOrEmptyValidator(minimum, maximum, self))
        elif dtype == float:
            self.setValidator(DoubleOrEmptyValidator(minimum, maximum, 5, self))
        else:
            raise ValueError("dtype must be int or float")

        self.setText(str(default))

    def getValue(self):
        if self.text().strip() == "":
            return None
        return self.dtype(self.text())
