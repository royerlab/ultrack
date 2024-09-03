import logging
import webbrowser
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Generator

import napari
import qtawesome as qta
import toml
from magicgui.widgets import create_widget
from napari.layers import Image, Layer
from napari.qt.threading import thread_worker
from pydantic import ValidationError
from qtpy.QtCore import Qt
from qtpy.QtGui import QCursor, QFont
from qtpy.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpacerItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from ultrack import MainConfig, export_tracks_by_extension
from ultrack.widgets.ultrackwidget.components.button_workflow_config import (
    ButtonWorkflowConfig,
)
from ultrack.widgets.ultrackwidget.components.emitting_stream import EmittingStream
from ultrack.widgets.ultrackwidget.data_forms import DataForms
from ultrack.widgets.ultrackwidget.utils import UltrackInput
from ultrack.widgets.ultrackwidget.workflows import (
    UltrackWorkflow,
    WorkflowChoice,
    WorkflowStage,
)

LOG = logging.getLogger(__name__)


def _create_link_buttons() -> list[QPushButton]:
    """
    Create buttons for external links.

    Returns
    -------
    List of QPushButton
        List of buttons with icons and link actions.
    """
    data = {
        ("Github", "mdi.github", "https://github.com/royerlab/ultrack"),
        ("BioArXiv", "mdi.file-document", ""),
        ("Documentation", "mdi.book-open", "https://royerlab.github.io/ultrack/"),
    }

    bts = []
    for name, icon_name, link in data:
        bt = QPushButton(name)
        icon = qta.icon(icon_name)
        bt.setIcon(icon)
        bt.clicked.connect(lambda _, _link=link: webbrowser.open(_link))
        bts.append(bt)

    return bts


class UltrackWidget(QWidget):
    def __init__(self, viewer: napari.Viewer) -> None:
        """
        Initialize the UltrackWidgetV2.

        Parameters
        ----------
        viewer : napari.Viewer
            The napari viewer instance.
        """
        super().__init__()
        self.viewer = viewer
        self.workflow = UltrackWorkflow(viewer)
        self._cb_images = {}
        self._data_forms = DataForms(self._on_change_config)

        self._config_valid = True
        self._all_images_are_valid = False

        self._init_ui()
        self._setup_signals()

        # Set the default workflow
        self._cb_workflow.setCurrentIndex(0)
        self._cb_workflow.currentIndexChanged.emit(0)

        self._on_image_changed(-1)
        self._on_layers_change(None)

        self._current_worker = None

    def _init_ui(self) -> None:
        """
        Initialize the user interface components.
        """
        layout = QVBoxLayout()
        self.setLayout(layout)

        self._add_title(layout)
        self._add_link_buttons(layout)
        self._add_spacer(layout, 10)
        self._add_configuration_group(layout)
        self._add_validation_messages_group(layout)
        self._add_run_button(layout)
        self._add_output_area(layout)
        self._add_cancel_button(layout)
        self._add_bt_export_tracks(layout)

        layout.addStretch()

    def _add_validation_messages_group(self, layout: QVBoxLayout) -> None:
        """
        Add the validation messages group to the layout.

        Parameters
        ----------
        layout : QVBoxLayout
            The layout to which the validation messages group will be added.
        """
        self._validation = QGroupBox("Validation messages")
        self._validation.setStyleSheet("QGroupBox { font-weight: bold; color: red;}")
        self._validation_messages = QLabel()
        self._validation_messages.setWordWrap(True)
        self._validation.hide()
        self._validation_messages.setStyleSheet("font-weight: normal; color: red")
        self._validation.setLayout(QVBoxLayout())
        self._validation.layout().addWidget(self._validation_messages)
        layout.addWidget(self._validation)

    def _add_bt_export_tracks(self, layout: QVBoxLayout) -> None:
        """
        Add the export tracks button to the layout.

        Parameters
        ----------
        layout : QVBoxLayout
            The layout to which the export tracks button will be added.
        """
        self._add_spacer(layout, 10)
        self._bt_export = QPushButton("Export tracks")
        self._bt_export.setEnabled(False)
        layout.addWidget(self._bt_export)

    def export_tracks(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.AnyFile)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter(
            "Napari Tracks (*.csv);;"
            "Trackmate (*.xml);;"
            "Zarr segments (*.zarr);;"
            "NetworkX (*.dot);;"
            "NetworkX (*.json)"
        )

        if file_dialog.exec_():
            file_name = file_dialog.selectedFiles()[0]
            ext = file_dialog.selectedNameFilter().split("*.")[-1][:-1]

            # add the extension if not present
            if not file_name.endswith(ext):
                file_name += f".{ext}"

            config = self._data_forms.get_config()
            export_tracks_by_extension(config, file_name, overwrite=True)

            QMessageBox.information(
                self,
                "Export tracks",
                f"Tracks exported to {file_name}",
                QMessageBox.Ok,
            )

    def _add_title(self, layout: QVBoxLayout) -> None:
        """
        Add the title and subtitle to the layout.

        Parameters
        ----------
        layout : QVBoxLayout
            The layout to which the title and subtitle will be added.
        """
        layout.addWidget(QLabel("<h1>ultrack</h1>"))
        subtitle = QLabel(
            "<h4>Large-scale cell tracking under segmentation uncertainty</h4>"
        )
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

    def _add_link_buttons(self, layout: QVBoxLayout) -> None:
        """
        Add link buttons to the layout.

        Parameters
        ----------
        layout : QVBoxLayout
            The layout to which the link buttons will be added.
        """
        layout_bt_links = QHBoxLayout()
        for bt in _create_link_buttons():
            layout_bt_links.addWidget(bt)
        layout.addLayout(layout_bt_links)

    def _add_spacer(self, layout: QVBoxLayout, size: int = 10) -> None:
        """
        Add a spacer item to the layout.

        Parameters
        ----------
        layout : QVBoxLayout
            The layout to which the spacer will be added.
        size : int
            The size of the spacer.
        """
        layout.addSpacerItem(QSpacerItem(0, size))

    def _add_configuration_group(self, layout: QVBoxLayout) -> None:
        """
        Add the configuration group to the layout.

        Parameters
        ----------
        layout : QVBoxLayout
            The layout to which the configuration group will be added.
        """
        gp_ultrack = QGroupBox("Ultrack Configuration")
        font = QFont()
        font.setBold(True)
        gp_ultrack.setFont(font)
        gp_layout = QVBoxLayout()
        gp_ultrack.setLayout(gp_layout)

        # Description
        gp_layout.addSpacerItem(QSpacerItem(0, 5))
        self._add_configuration_description(gp_layout)

        # Workflow selector
        gp_layout.addSpacerItem(QSpacerItem(0, 10))
        self._add_workflow_selector(gp_layout)

        # Image selectors
        self._add_image_selectors(gp_layout)
        gp_layout.addSpacerItem(QSpacerItem(0, 10))

        # Settings buttons
        self._add_settings_buttons(gp_layout)

        # Form tabs
        self._add_form_tabs(gp_layout)

        layout.addSpacerItem(QSpacerItem(0, 5))
        layout.addWidget(gp_ultrack)

        self.main_group = gp_ultrack

    def _add_configuration_description(self, layout: QVBoxLayout) -> None:
        """
        Add the configuration description to the layout.

        Parameters
        ----------
        layout : QVBoxLayout
            The layout to which the configuration description will be added.
        """
        label = QLabel(
            "Configure the Ultrack parameters and run the tracking algorithm."
        )
        label.setWordWrap(True)
        layout.addWidget(label)

    def _add_workflow_selector(self, layout: QVBoxLayout) -> None:
        """
        Add the workflow selector to the layout.

        Parameters
        ----------
        layout : QVBoxLayout
            The layout to which the workflow selector will be added.
        """
        layout.addWidget(QLabel("<h5>Select the desired Ultrack workflow</h5>"))
        self._cb_workflow = QComboBox()
        for workflow_choice in WorkflowChoice:
            name = workflow_choice.value
            self._cb_workflow.addItem(name, workflow_choice)
        layout.addWidget(self._cb_workflow)

    def _add_image_selectors(self, layout: QVBoxLayout) -> None:
        """
        Add image selectors to the layout.

        Parameters
        ----------
        layout : QVBoxLayout
            The layout to which the image selectors will be added.
        """
        self.images = ["image", "contours", "detection", "labels"]

        for image in UltrackInput:
            image_desc = image.value
            label = QLabel(image_desc)
            layout.addWidget(label)
            widget = create_widget(annotation=Image, label=image_desc)
            widget._lb = label
            widget.native.setPlaceholderText("Select image")
            self._cb_images[image] = widget
            layout.addWidget(widget.native)

    def _add_settings_buttons(self, layout: QVBoxLayout) -> None:
        """
        Add settings buttons to the layout.

        Parameters
        ----------
        layout : QVBoxLayout
            The layout to which the settings buttons will be added.
        """
        button_layout = QHBoxLayout()
        self._bt_toggle_settings = self._create_toggle_settings_button()
        button_layout.addWidget(self._bt_toggle_settings)

        self._bt_load_settings = self._create_button(
            "mdi.folder-open", "Load settings from file"
        )
        button_layout.addWidget(self._bt_load_settings)

        self._bt_save_settings = self._create_button(
            "mdi.content-save", "Save settings to file"
        )
        button_layout.addWidget(self._bt_save_settings)

        layout.addLayout(button_layout)

    def _create_toggle_settings_button(self) -> QPushButton:
        """
        Create the toggle settings button.

        Returns
        -------
        QPushButton
            The created toggle settings button.
        """
        button = QPushButton("Show advanced Settings")
        button.setCheckable(True)
        button.setChecked(False)
        button.setIcon(qta.icon("mdi.chevron-down"))
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        return button

    def _create_button(self, icon_name: str, tooltip: str) -> QPushButton:
        """
        Create a button with an icon and tooltip.

        Parameters
        ----------
        icon_name : str
            The name of the icon.
        tooltip : str
            The tooltip text.

        Returns
        -------
        QPushButton
            The created button.
        """
        button = QPushButton(qta.icon(icon_name), "")
        button.setToolTip(tooltip)
        return button

    def _add_form_tabs(self, layout: QVBoxLayout) -> None:
        """
        Add form tabs to the layout.

        Parameters
        ----------
        layout : QVBoxLayout
            The layout to which the form tabs will be added.
        """
        self._tab_scroll_area = QScrollArea()
        self._tab_scroll_area.setWidgetResizable(True)
        self._tab_scroll_area.setWidget(self._data_forms.get_tab_widget())
        self._tab_scroll_area.hide()
        layout.addWidget(self._tab_scroll_area)

    def _add_run_button(self, layout: QVBoxLayout) -> None:
        """
        Add the run button to the layout.

        Parameters
        ----------
        layout : QVBoxLayout
            The layout to which the run button will be added.
        """
        self._bt_run = QPushButton(self)
        self._bt_run.setText("Run")
        self._bt_run.setEnabled(False)
        self._bt_run.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)

        button_layout = QHBoxLayout()
        button_layout.addWidget(self._bt_run)

        self._bt_run_config = ButtonWorkflowConfig(self, self.workflow)
        button_layout.addWidget(self._bt_run_config)

        layout.addLayout(button_layout)

    def _setup_signals(self) -> None:
        """Set up the signals for various buttons and widgets."""
        self._bt_toggle_settings.clicked.connect(self._on_toggle_settings)
        self._bt_save_settings.clicked.connect(self._on_save_settings)
        self._bt_load_settings.clicked.connect(self._on_load_settings)
        self._bt_export.clicked.connect(self.export_tracks)
        self._bt_run.clicked.connect(self._on_run)
        self._bt_cancel.clicked.connect(self._cancel)
        self._cb_workflow.currentIndexChanged.connect(self._on_workflow_changed)
        self.viewer.layers.events.removed.connect(self._on_layers_change)
        self.viewer.layers.events.inserted.connect(self._on_layers_change)

        for cb_image in self._cb_images.values():
            cb_image.changed.connect(self._on_image_changed)

    def _on_change_config(self):
        """Handle the change of the configuration."""
        if hasattr(self, "_bt_run"):
            try:
                new_config = self._data_forms.get_config()
                additional_config = self._data_forms.get_additional_options()
                inputs = {k: w.value for k, w in self._cb_images.items()}
                workflow = self.workflow.get_stage(
                    new_config, additional_options=additional_config, inputs=inputs
                )
                self._bt_run_config.set_workflow_stage(workflow)
                self._bt_run.setEnabled(self._all_images_are_valid)
                self._validation.hide()
                self._config_valid = True
            except ValidationError as e:
                msg = "<ul>"
                for err in e.errors():
                    msg += f"<li>{err['msg']}.</li>"
                msg += "</ul>"
                self._validation_messages.setText(msg)
                self._validation.show()
                self._bt_run.setEnabled(False)
                self._config_valid = False

    def _on_run(self):
        """Handle the run button click event.

        It runs the selected workflow with the provided inputs.
        """
        config = self._data_forms.get_config()
        additional_config = self._data_forms.get_additional_options()
        workflow_choice = self._cb_workflow.currentData()

        inputs: dict[UltrackInput, Layer] = {}

        for k, w in self._cb_images.items():
            if w.value is not None:
                inputs[k] = w.value

        worker = self._make_run_worker(
            config, workflow_choice, inputs, additional_options=additional_config
        )
        self._current_worker = worker
        worker.started.connect(self._on_run_started)
        worker.yielded.connect(self._on_workflow_intermediate_results)
        worker.finished.connect(self._on_run_finished)
        worker.start()

    def _on_workflow_intermediate_results(self, layer: Layer):
        """Handle the intermediate results of the workflow."""
        self.viewer.add_layer(layer)

    def _on_run_started(self):
        """Handle the start of the run worker."""
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        self._bt_run.setEnabled(False)
        self._bt_export.setEnabled(False)
        self._bt_run.setText("Running...")
        self._bt_run.repaint()
        self.ctx_stdout_switcher.__enter__()
        self.ctx_stderr_switcher.__enter__()
        self._output.show()
        self._bt_cancel.show()
        self._bt_cancel.setEnabled(True)
        self.main_group.setEnabled(False)

    def _on_run_finished(self):
        """Handle the finish of the run worker."""
        self._bt_run.setEnabled(True)
        self._bt_run.setText("Run")
        self._bt_export.setEnabled(
            self.workflow.last_reached_stage == WorkflowStage.DONE
        )
        self._bt_run.repaint()
        self.ctx_stdout_switcher.__exit__(None, None, None)
        self.ctx_stderr_switcher.__exit__(None, None, None)
        self._output.hide()
        self._bt_cancel.hide()
        self.main_group.setEnabled(True)
        self._current_worker = None
        QApplication.restoreOverrideCursor()

    @thread_worker
    def _make_run_worker(
        self,
        config: MainConfig,
        workflow_choice: WorkflowChoice,
        inputs: dict[UltrackInput, Layer],
        additional_options: dict[Any],
    ) -> Generator:
        """
        Create a worker to run the selected workflow.

        Returns
        -------
        Worker
            The worker to run the selected workflow.
        """
        yield from self.workflow.run(
            config, workflow_choice, inputs, additional_options
        )

    def _on_save_settings(self) -> None:
        """Handle the save settings button click event."""
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            None,
            "Save TOML",
            "ultrack_settings.toml",
            "TOML Files (*.toml);;All Files (*)",
            options=options,
        )
        if file_name:
            with open(file_name, "w") as f:
                data = self._data_forms.get_config().dict(by_alias=True)
                toml.dump(data, f)
            print(f"Data saved to {file_name}")

    def _on_load_settings(self) -> None:
        """
        Handle the load settings button click event.
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            None, "Open TOML", "", "TOML Files (*.toml);;All Files (*)", options=options
        )
        if file_name:
            with open(file_name) as f:
                data = toml.load(f)
                data = MainConfig(**data)
                self._data_forms.load_config(data)

    def _on_toggle_settings(self):
        """
        Handle the toggle settings button click event.
        """
        if self._bt_toggle_settings.isChecked():
            self._tab_scroll_area.show()
            icon = qta.icon("mdi.chevron-up")
            self._bt_toggle_settings.setIcon(icon)
            self._bt_toggle_settings.setText("Hide advanced Settings")
        else:
            self._tab_scroll_area.hide()
            icon = qta.icon("mdi.chevron-down")
            self._bt_toggle_settings.setIcon(icon)
            self._bt_toggle_settings.setText("Show advanced Settings")

    def _on_layers_change(self, _: Any) -> None:
        """
        Update layer choices when layers are changed.
        """
        for widget in self._cb_images.values():
            widget.choices = self.viewer.layers

    def _on_image_changed(self, _: int) -> None:
        """
        Handle image selection changes.

        Parameters
        ----------
        _ : int
            Ignored.
        """
        self._all_images_are_valid = True

        for ultrack_input, widget in self._cb_images.items():
            if widget.enabled:
                if widget.value is None:
                    self._all_images_are_valid = False
                if ultrack_input == UltrackInput.IMAGE:
                    self._data_forms.notify_image_update(widget.value)

        self._bt_run.setEnabled(self._all_images_are_valid and self._config_valid)

    def _on_workflow_changed(self, index: int) -> None:
        """
        Handle workflow selection changes.

        Parameters
        ----------
        index : int
            The index of the selected workflow.
        """
        workflow_choice = self._cb_workflow.itemData(index)
        inputs = self.workflow.inputs_from_choice(workflow_choice)
        for name, widget in self._cb_images.items():
            if name in inputs:
                widget.native.show()
                widget.native.setEnabled(True)
                widget._lb.show()
            else:
                widget.native.hide()
                widget.native.setEnabled(False)
                widget._lb.hide()

        self._data_forms.setup_additional_options(workflow_choice)

    def _add_output_area(self, layout: QVBoxLayout) -> None:
        """
        Add the output area to the layout.

        Parameters
        ----------
        layout : QVBoxLayout
            The layout to which the output area will be added.
        """
        self._output = QTextEdit()
        self._output.setReadOnly(True)
        self._output.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self._output.hide()
        self.ctx_stdout_switcher = redirect_stdout(EmittingStream(self._output, "gray"))
        self.ctx_stderr_switcher = redirect_stderr(EmittingStream(self._output, "red"))
        layout.addWidget(self._output)

    def _add_cancel_button(self, layout: QVBoxLayout) -> None:
        """
        Add the cancel button to the layout.

        Parameters
        ----------
        layout : QVBoxLayout
            The layout to which the cancel button will be added.
        """
        self._bt_cancel = QPushButton(self)
        self._bt_cancel.hide()
        self._bt_cancel.setText("Cancel")
        self._bt_cancel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        layout.addWidget(self._bt_cancel)

    def _cancel(self):
        """Cancel the current worker."""
        if self._current_worker is not None:
            self._current_worker.quit()
            self._bt_cancel.setEnabled(False)


if __name__ == "__main__":
    from napari import Viewer

    viewer = Viewer()
    widget = UltrackWidget(viewer)
    widget.show()
    viewer.window.add_dock_widget(widget, area="right")
    napari.run()
