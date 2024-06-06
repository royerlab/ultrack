import time
from pathlib import Path
from typing import Callable, List

import napari
import numpy as np
import pytest

from ultrack.config import MainConfig
from ultrack.widgets.ultrackwidget import UltrackWidget
from ultrack.widgets.ultrackwidget.utils import UltrackInput
from ultrack.widgets.ultrackwidget.workflows import WorkflowChoice


@pytest.mark.parametrize(
    "workflow",
    [
        WorkflowChoice.AUTO_DETECT,
        WorkflowChoice.MANUAL,
        WorkflowChoice.AUTO_FROM_LABELS,
    ],
)
@pytest.mark.parametrize("multichannel", [True, False])
def test_ultrack_widget(
    make_napari_viewer: Callable[[], napari.Viewer],
    zarr_dataset_paths: List[str],
    workflow: WorkflowChoice,
    tmp_path: Path,
    multichannel: bool,
) -> None:
    if multichannel and workflow != WorkflowChoice.AUTO_DETECT:
        pytest.skip("Multichannel only supported for auto-detect workflow")

    viewer = make_napari_viewer()
    widget = UltrackWidget(viewer)

    layers = viewer.open(zarr_dataset_paths)  # foreground, edge, label

    if multichannel:
        data = np.asarray(layers[0].data, dtype=float)
        image_data = np.stack([data, data, data], axis=-1)
    else:
        image_data = np.asarray(layers[0].data, dtype=float)

    viewer.add_image(image_data, name="image")

    layers = viewer.layers

    workflow_idx = widget._cb_workflow.findData(workflow)
    widget._cb_workflow.setCurrentIndex(workflow_idx)
    widget._cb_workflow.currentIndexChanged.emit(workflow_idx)

    # setting combobox choices manually, because they were not working automatically
    widget._cb_images[UltrackInput.DETECTION].choices = layers
    widget._cb_images[UltrackInput.CONTOURS].choices = layers
    widget._cb_images[UltrackInput.LABELS].choices = layers
    widget._cb_images[UltrackInput.IMAGE].choices = layers

    # selecting layers
    widget._cb_images[UltrackInput.DETECTION].value = layers[0]
    widget._cb_images[UltrackInput.CONTOURS].value = layers[1]
    widget._cb_images[UltrackInput.LABELS].value = layers[2]
    widget._cb_images[UltrackInput.IMAGE].value = layers[3]

    ####################################################################################
    # CHECK IF THE CHANNEL SELECTION IS PROPERLY PROPAGATED
    ####################################################################################
    if workflow == WorkflowChoice.AUTO_DETECT:
        additional_options = widget._data_forms.get_additional_options()
        expected_channel_value = 3 if multichannel else None
        assert (
            additional_options["robust_invert_kwargs"]["channel_axis"]
            == expected_channel_value
        )
        assert (
            additional_options["detect_foreground_kwargs"]["channel_axis"]
            == expected_channel_value
        )

    ####################################################################################
    # CHECK IF THE CONFIGURATION IS PROPERLY PROPAGATED
    ####################################################################################
    config = MainConfig()
    config.segmentation_config.threshold = 0.65

    # temporary working dir
    config.data_config.working_dir = tmp_path

    widget._data_forms.load_config(config)

    propagated_config = widget._data_forms.get_config()

    assert propagated_config.segmentation_config.threshold == 0.65

    ####################################################################################
    # CHECK IF THE WIDGET CAN RUN
    ####################################################################################
    widget._bt_run.clicked.emit()

    time.sleep(1)
    while widget._current_worker is not None and widget._current_worker.is_running:
        time.sleep(0.5)
