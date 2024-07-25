import json
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import qtawesome as qta
from napari.layers import Image, Layer
from qtpy.QtWidgets import (
    QCheckBox,
    QGroupBox,
    QLabel,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ultrack import MainConfig
from ultrack.widgets.ultrackwidget.components.blankable_number_edit import (
    BlankableNumberEdit,
)
from ultrack.widgets.ultrackwidget.components.no_wheel_spin import (
    NoWheelDoubleSpinBox,
    NoWheelSpinBox,
)
from ultrack.widgets.ultrackwidget.workflows import WorkflowChoice


class DataForms:
    """
    A class to manage and display data forms for the Ultrack configuration.

    Attributes
    ----------
    _tab : QTabWidget
        The main tab widget containing all forms.
    _bindings : list
        Bindings for standard form fields.
    _additional_bindings : list
        Bindings for additional form fields.
    _update_channel_axis_bindings : list
        Bindings for updating the channel axis field based on image data.
    _additional_forms : dict
        Additional forms for optional configuration sections.

    Methods
    -------
    _setup_forms()
        Setup and initialize the forms based on metadata.
    load_config(config: MainConfig)
        Load the configuration into the form fields.
    _create_form(id_form: str, metadata: Dict[str, Any])
        Create a form based on provided metadata.
    _create_field(layout: QVBoxLayout, id_form: str, id_field: str, field_metadata: Dict, current_subform: str)
        Create an individual form field.
    _create_checkbox_field(layout: QVBoxLayout, id_form: str, id_field: str, field_metadata: Dict, current_subform: str)
        Create a checkbox field.
    _create_number_field(layout: QVBoxLayout, id_form: str, id_field: str, field_metadata: Dict, current_subform: str)
        Create a number input field.
    get_additional_options() -> Dict[str, Any]
        Get the additional options from the form.
    get_config() -> MainConfig
        Get the current configuration from the form.
    setup_additional_options(additional_options: List[str])
        Setup visibility of additional options based on input.
    get_tab_widget() -> QTabWidget
        Get the main tab widget.
    """

    def __init__(self, on_change_any_input: Callable) -> None:
        """
        Initialize the DataForms class.

        Parameters
        ----------
        on_change_any_input : Callable
            Callback function to be called when any input changes.
        """
        self.on_change_any_input = on_change_any_input

        self._config = None
        self._tab = QTabWidget()
        self._bindings = []
        self._update_channel_axis_bindings: list[BlankableNumberEdit] = []
        self._additional_bindings = []
        self._additional_forms = {}
        self._setup_forms()
        self.load_config(MainConfig())

    def _setup_forms(self) -> None:
        """
        Setup and initialize the forms based on metadata.
        """
        rsrc_dir = Path(__file__).parent / "resources"
        with open(rsrc_dir / "forms.json") as forms, open(
            rsrc_dir / "additional_options.json"
        ) as opts:
            forms_metadata = json.load(forms)
            additional_options = json.load(opts)
            forms_metadata["additional_options"] = {
                "title": "Pre-processing",
                "fields": additional_options,
            }

        self._create_form("additional_options", forms_metadata["additional_options"])
        self._create_form("segmentation", forms_metadata["segmentation"])
        self._create_form("linking", forms_metadata["linking"])
        self._create_form("tracking", forms_metadata["tracking"])

    def load_config(self, config: MainConfig) -> None:
        """
        Load the configuration into the form fields.

        Parameters
        ----------
        config : MainConfig
            The main configuration to load.
        """
        self._config = config.dict(by_alias=True)
        for id_form, id_field, widget, getter, setter in self._bindings:
            value = self._config[id_form][id_field]
            getattr(widget, setter)(value)

    def _create_form(self, id_form: str, metadata: Dict[str, Any]) -> None:
        """
        Create a form based on provided metadata.

        Parameters
        ----------
        id_form : str
            The form identifier.
        metadata : Dict[str, Any]
            Metadata describing the form fields and layout.
        """
        title = metadata["title"]
        fields = metadata["fields"]
        tab = QWidget()
        layout = QVBoxLayout()
        main_layout = layout
        tab.setLayout(layout)

        fields_to_be_added = [*fields.items()]
        current_subform = None
        while fields_to_be_added:
            id_field, field_metadata = fields_to_be_added.pop(0)

            if not field_metadata.get("subform", False):
                self._create_field(
                    layout, id_form, id_field, field_metadata, current_subform
                )
            else:
                subgroup = QGroupBox(field_metadata["title"])
                current_subform = id_field
                layout = QVBoxLayout()
                subgroup.setLayout(layout)
                fields_to_be_added = [
                    *field_metadata["fields"].items()
                ] + fields_to_be_added
                main_layout.addWidget(subgroup)
                self._additional_forms[id_field] = subgroup

        main_layout.addStretch()
        self._tab.addTab(tab, title)

    def _create_field(
        self,
        layout: QVBoxLayout,
        id_form: str,
        id_field: str,
        field_metadata: Dict,
        current_subform: Optional[str],
    ) -> None:
        """
        Create an individual form field.

        Parameters
        ----------
        layout : QVBoxLayout
            The layout to which the field will be added.
        id_form : str
            The form identifier.
        id_field : str
            The field identifier.
        field_metadata : Dict
            Metadata describing the field.
        current_subform : Optional[str]
            The current subform identifier if within a subform.
        """
        if field_metadata["type"] == "number":
            label = QLabel(f'{field_metadata["label"]} {chr(0xF059)}')
            label.setFont(qta.font("fa", 14))
            label.setToolTip(field_metadata["tooltip"])
            layout.addWidget(label)

            self._create_number_field(
                layout, id_form, id_field, field_metadata, current_subform
            )
        elif field_metadata["type"] == "checkbox":
            self._create_checkbox_field(
                layout, id_form, id_field, field_metadata, current_subform
            )
        else:
            raise NotImplementedError(
                f"Field type {field_metadata['type']} not implemented"
            )

    def _create_checkbox_field(
        self,
        layout: QVBoxLayout,
        id_form: str,
        id_field: str,
        field_metadata: Dict,
        current_subform: Optional[str],
    ) -> None:
        """
        Create a checkbox field.

        Parameters
        ----------
        layout : QVBoxLayout
            The layout to which the field will be added.
        id_form : str
            The form identifier.
        id_field : str
            The field identifier.
        field_metadata : Dict
            Metadata describing the field.
        current_subform : Optional[str]
            The current subform identifier if within a subform.
        """
        checkbox = QCheckBox(f'{field_metadata["label"]} {chr(0xF059)}')
        checkbox.setToolTip(field_metadata["tooltip"])
        checkbox.setChecked(bool(field_metadata.get("default", False)))
        checkbox.stateChanged.connect(self.on_change_any_input)
        layout.addWidget(checkbox)
        if id_form != "additional_options":
            self._bindings.append(
                (id_form, id_field, checkbox, "isChecked", "setChecked")
            )
        else:
            self._additional_bindings.append(
                (current_subform, id_field, checkbox, "isChecked")
            )

        if id_field == "__enable__":
            checkbox.stateChanged.connect(
                lambda state, subform=current_subform, _checkbox=checkbox: self._disable_subform(
                    state, self._additional_forms[subform], _checkbox
                )
            )

    def _disable_subform(
        self, state: bool, subform: QGroupBox, caller: QCheckBox
    ) -> None:
        """Disable all widgets in a subform except the caller.

        Parameters
        ----------
        state : bool
            The state to set the widgets to.
        subform : QGroupBox
            The subform to disable.
        caller : QCheckBox
            The caller widget that should remain enabled.
        """
        for widget in subform.children():
            if widget != caller:
                widget.setEnabled(state)

    def _create_number_field(
        self,
        layout: QVBoxLayout,
        id_form: str,
        id_field: str,
        field_metadata: Dict,
        current_subform: Optional[str],
    ) -> None:
        """
        Create a number input field.

        Parameters
        ----------
        layout : QVBoxLayout
            The layout to which the field will be added.
        id_form : str
            The form identifier.
        id_field : str
            The field identifier.
        field_metadata : Dict
            Metadata describing the field.
        current_subform : Optional[str]
            The current subform identifier if within a subform.
        """
        dtype = float if field_metadata.get("step") == "any" else int
        max_value = dtype(field_metadata.get("max", 999999999))
        min_value = dtype(field_metadata.get("min", -999999999))
        default = field_metadata.get("default", 0)

        if field_metadata.get("required", False):
            spin_box = NoWheelDoubleSpinBox() if dtype == float else NoWheelSpinBox()
            if dtype == float:
                spin_box.setDecimals(5)
            spin_box.setMinimum(min_value)
            spin_box.setMaximum(max_value)
            spin_box.setValue(dtype(default))
            layout.addWidget(spin_box)
            spin_box.valueChanged.connect(self.on_change_any_input)

            if id_form != "additional_options":
                self._bindings.append(
                    (id_form, id_field, spin_box, "value", "setValue")
                )
            else:
                self._additional_bindings.append(
                    (current_subform, id_field, spin_box, "value")
                )
        else:
            number_edit = BlankableNumberEdit(
                default=default,
                dtype=dtype,
                minimum=min_value,
                maximum=max_value,
            )
            layout.addWidget(number_edit)
            number_edit.textChanged.connect(self.on_change_any_input)

            if id_field == "channel_axis":
                self._update_channel_axis_bindings.append(number_edit)

            if id_form != "additional_options":
                self._bindings.append(
                    (
                        id_form,
                        id_field,
                        number_edit,
                        "getValue",
                        "setText",
                    )
                )
            else:
                self._additional_bindings.append(
                    (
                        current_subform,
                        id_field,
                        number_edit,
                        "getValue",
                    )
                )

    def get_additional_options(self) -> Dict[str, Any]:
        """
        Get the additional options from the form.

        Returns
        -------
        Dict[str, Any]
            A dictionary of additional options.
        """
        additional_options = {}
        for id_form, id_field, widget, getter in self._additional_bindings:
            if not self._additional_forms[id_form].isHidden():
                value = getattr(widget, getter)()
                if id_form not in additional_options:
                    additional_options[id_form] = {}
                additional_options[id_form][id_field] = value
        return additional_options

    def get_config(self) -> MainConfig:
        """
        Get the current configuration from the form.

        Returns
        -------
        MainConfig
            The main configuration object.
        """
        for id_form, id_field, widget, getter, setter in self._bindings:
            value = getattr(widget, getter)()
            self._config[id_form][id_field] = value
        return MainConfig.parse_obj(self._config)

    def setup_additional_options(self, workflow_choice: WorkflowChoice) -> None:
        """
        Setup visibility of additional options based on input.

        Each workflow choice has a different set of additional options. This method
        will show or hide the additional options based on the workflow choice.
        The additional options are contained in the field `_additional_forms` and
        maps the form id to the form widget. Since the workflow maps to the form id,
        we can show or hide the form based on the workflow choice.

        Parameters
        ----------
        workflow_choice : str
            The workflow choice to determine which additional options to show.
        """
        map_workflow_to_form_ids = {
            WorkflowChoice.AUTO_FROM_LABELS: ["label_to_contours_kwargs"],
            WorkflowChoice.AUTO_DETECT: [
                "detect_foreground_kwargs",
                "robust_invert_kwargs",
                "flow_kwargs",
            ],
            WorkflowChoice.MANUAL: ["flow_kwargs"],
        }

        any_visible = False
        for id_options in self._additional_forms.keys():
            if id_options in map_workflow_to_form_ids[workflow_choice]:
                any_visible = True
                self._additional_forms[id_options].show()
            else:
                self._additional_forms[id_options].hide()

        # hard-coded index for additional options tab
        # if no additional options are visible, hide the tab
        self._tab.setTabVisible(0, any_visible)

    def notify_image_update(self, image: Layer) -> None:
        channel = ""
        if image and isinstance(image, Image) and image.rgb:
            channel = image.data.ndim - 2  # last dimension is channel

        for widget in self._update_channel_axis_bindings:
            widget.setText(str(channel))

    def get_tab_widget(self) -> QTabWidget:
        """
        Get the main tab widget.

        Returns
        -------
        QTabWidget
            The main tab widget containing all forms.
        """
        return self._tab
