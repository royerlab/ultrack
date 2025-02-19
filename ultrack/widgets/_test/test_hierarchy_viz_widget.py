import time
from typing import Callable, Tuple

import napari
import pytest
import zarr
from sqlalchemy import Column

from ultrack.config import MainConfig
from ultrack.core.database import NodeDB
from ultrack.widgets.hierarchy_viz_widget import HierarchyVizWidget
from ultrack.widgets.ultrackwidget import UltrackWidget
from ultrack.widgets.ultrackwidget.utils import UltrackInput
from ultrack.widgets.ultrackwidget.workflows import WorkflowChoice


@pytest.mark.parametrize(
    "node_attribute", [NodeDB.id, NodeDB.selected, NodeDB.z, NodeDB.node_annot]
)
def test_hierarchy_viz_widget_from_config(
    make_napari_viewer: Callable[[], napari.Viewer],
    segmentation_database_mock_data: MainConfig,
    node_attribute: Column,
) -> None:
    config = segmentation_database_mock_data
    viewer = make_napari_viewer()
    widget = HierarchyVizWidget(viewer, config, node_attribute=node_attribute)
    viewer.window.add_dock_widget(widget)

    assert HierarchyVizWidget.HIER_LAYER_NAME in viewer.layers

    # test moving sliders:
    widget._slider_update(0.75)
    widget._slider_update(0.25)

    # test is shape of layer.data has same shape as the data shape reported in config:
    assert (
        tuple(config.data_config.metadata["shape"])
        == viewer.layers[HierarchyVizWidget.HIER_LAYER_NAME].data.shape
    )  # metadata["shape"] is a list, data.shape is a tuple

    # checking that there are some labels painted
    assert viewer.layers[HierarchyVizWidget.HIER_LAYER_NAME].data[0].max() > 0


def test_hierarchy_viz_widget_from_ultrack_widget(
    make_napari_viewer: Callable[[], napari.Viewer],
    timelapse_mock_data: Tuple[zarr.Array, zarr.Array, zarr.Array],
    request,
) -> None:
    # make napari viewer
    viewer = make_napari_viewer()

    # get mock segmentation data + add to viewer
    segments = timelapse_mock_data[2]
    viewer.add_labels(segments, name="segments")

    # open ultrack widget
    ultrack_widget = UltrackWidget(viewer)
    viewer.window.add_dock_widget(ultrack_widget)

    # setup ultrack widget for 'Labels' input
    layers = viewer.layers
    workflow = WorkflowChoice.AUTO_FROM_LABELS
    workflow_idx = ultrack_widget._cb_workflow.findData(workflow)
    ultrack_widget._cb_workflow.setCurrentIndex(workflow_idx)
    ultrack_widget._cb_workflow.currentIndexChanged.emit(workflow_idx)
    # setting combobox choices manually, because they were not working automatically
    ultrack_widget._cb_images[UltrackInput.LABELS].choices = layers
    # # selecting layers
    ultrack_widget._cb_images[UltrackInput.LABELS].value = layers["segments"]
    ultrack_widget._bt_run.click()

    time.sleep(1)
    # wait for run to finish before loading config
    while (
        ultrack_widget._current_worker is not None
        and ultrack_widget._current_worker.is_running
    ):
        time.sleep(0.5)

    hier_viz_widget = HierarchyVizWidget(viewer)
    viewer.window.add_dock_widget(hier_viz_widget)

    assert HierarchyVizWidget.HIER_LAYER_NAME in viewer.layers

    # test moving sliders:
    hier_viz_widget._slider_update(0.75)
    hier_viz_widget._slider_update(0.25)

    if request.config.getoption("--show-napari-viewer"):
        napari.run()

    # test is shape of layer.data has same shape as the data shape reported in config:
    assert (
        tuple(ultrack_widget._data_forms.get_config().data_config.metadata["shape"])
        == viewer.layers[HierarchyVizWidget.HIER_LAYER_NAME].data.shape
    )  # metadata["shape"] is a list, data.shape in layer is a tuple

    # checking that there are some labels painted
    assert viewer.layers[HierarchyVizWidget.HIER_LAYER_NAME].data[0].max() > 0
