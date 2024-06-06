import qtawesome as qta
from qtpy.QtWidgets import QAction, QMenu, QPushButton

from ultrack.widgets.ultrackwidget.workflows import UltrackWorkflow, WorkflowStage


class ButtonWorkflowConfig(QPushButton):
    def __init__(self, parent, workflow: UltrackWorkflow):
        super().__init__(parent)
        self.setIcon(qta.icon("mdi.cog"))
        self.workflow = workflow
        self._setup_menu()
        self._setup_signals()

    def _setup_menu(self):
        self.menu = QMenu(self)
        # run all
        action = QAction("All", self.menu)
        action.setCheckable(True)
        action.setChecked(True)
        self.action_all = action
        self.menu.addAction(action)

        for workflow in WorkflowStage:
            if workflow == WorkflowStage.DONE:
                continue
            action = QAction(workflow.value, self.menu)
            action.setCheckable(True)
            action.setChecked(False)
            action.setData(workflow)
            self.menu.addAction(action)

    def _on_check_all(self):
        checked = self.action_all.isChecked()
        for action in self.menu.actions():
            action.setChecked(checked)

    def _on_check_workflow(self):
        # checked = self.sender().isChecked()
        # workflow = self.sender().data()

        self.action_all.triggered.disconnect(self._on_check_all)
        self.action_all.setChecked(False)
        self.action_all.triggered.connect(self._on_check_all)

    def _setup_signals(self):
        self.action_all.triggered.connect(self._on_check_all)
        for action in self.menu.actions():
            if action != self.action_all:
                action.triggered.connect(self._on_check_workflow)

    def set_workflow_stage(self, stage: WorkflowStage):
        self.action_all.setChecked(False)
        for action in self.menu.actions():
            if action.data() == stage:
                action.setChecked(True)
                stage = stage.next()
            else:
                action.setChecked(False)

    def get_workflow_stages(self) -> list[WorkflowStage]:
        stages = []
        for action in self.menu.actions():
            if action.isChecked():
                stages.append(action.data())
        return stages

    def mousePressEvent(self, e):
        # self._bt_run_options.mapToGlobal(event.pos)
        self.menu.exec_(self.mapToGlobal(e.pos()))
        super().mousePressEvent(e)
