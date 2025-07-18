from qtpy.QtGui import QTextCursor
from qtpy.QtWidgets import QTextEdit


class EmittingStream:
    def __init__(self, text_edit: QTextEdit, color: str = "black"):
        self.text_edit = text_edit
        self.color = color

    def write(self, data):
        formatted = f'<span style="color:{self.color}">{data}</span>'
        self.text_edit.append(formatted)
        self.text_edit.moveCursor(QTextCursor.MoveOperation.End)
        self.text_edit.ensureCursorVisible()

    def flush(self):
        pass
