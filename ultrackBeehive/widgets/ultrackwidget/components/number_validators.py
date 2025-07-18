from qtpy.QtGui import QDoubleValidator, QIntValidator, QValidator


class IntOrEmptyValidator(QIntValidator):
    def __init__(self, minimum, maximum, parent=None):
        super().__init__(minimum, maximum, parent)

    def validate(self, input, pos):
        if input == "":
            return QIntValidator.Acceptable, input, pos
        return super().validate(input, pos)


class DoubleOrEmptyValidator(QDoubleValidator):
    def __init__(self, bottom, top, decimals, parent=None):
        super().__init__(bottom, top, decimals, parent)

    def validate(self, input, pos):
        if input == "":
            return QValidator.Acceptable, input, pos
        return super().validate(input, pos)
