from qtpy.QtWidgets import QDoubleSpinBox, QSpinBox


class NoWheelSpinMixin:
    def wheelEvent(self, event):
        event.ignore()


class NoWheelSpinBox(NoWheelSpinMixin, QSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)


class NoWheelDoubleSpinBox(NoWheelSpinMixin, QDoubleSpinBox):
    def __init__(self, parent=None):
        super().__init__(parent)
