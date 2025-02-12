from typing import Callable, Tuple

import napari
import numpy as np
import pytest
import zarr

from ultrack.config import MainConfig
from ultrack.widgets.ultrackwidget import UltrackWidget
from ultrack.widgets.hierarchy_viz_widget import HierarchyVizWidget
from ultrack.widgets.ultrackwidget.workflows import WorkflowChoice
from ultrack.widgets.ultrackwidget.utils import UltrackInput





def test_hierarchy_viz_widget(
        make_napari_viewer: Callable[[],napari.Viewer],
        segmentation_database_mock_data: MainConfig,
        timelapse_mock_data: Tuple[zarr.Array, zarr.Array, zarr.Array],
        request,
    ):

    ####################################################################################
    #OPTION 1: run widget using config    
    ####################################################################################
    config = segmentation_database_mock_data
    viewer = make_napari_viewer()
    widget = HierarchyVizWidget(viewer,config)
    viewer.window.add_dock_widget(widget)

    assert "hierarchy" in viewer.layers

    #test moving sliders:
    widget._slider_update(0.75)
    widget._slider_update(0.25)

    #test is shape of layer.data has same shape as the data shape reported in config:
    assert tuple(config.data_config.metadata["shape"]) == viewer.layers['hierarchy'].data.shape     #metadata["shape"] is a list, data.shape is a tuple


    ####################################################################################
    #OPTION 2: run widget by taking config from Ultrack-widget
    ####################################################################################
    #make napari viewer
    viewer2 = make_napari_viewer()

    #get mock segmentation data + add to viewer
    segments = timelapse_mock_data[2]
    print('segments shape',segments.shape)
    viewer2.add_labels(segments,name='segments')

    #open ultrack widget
    widget_ultrack = UltrackWidget(viewer2)
    viewer2.window.add_dock_widget(widget_ultrack)

    #setup ultrack widget for 'Labels' input
    layers = viewer2.layers
    workflow = WorkflowChoice.AUTO_FROM_LABELS
    workflow_idx = widget_ultrack._cb_workflow.findData(workflow)
    widget_ultrack._cb_workflow.setCurrentIndex(workflow_idx)
    widget_ultrack._cb_workflow.currentIndexChanged.emit(workflow_idx)
    # setting combobox choices manually, because they were not working automatically
    widget_ultrack._cb_images[UltrackInput.LABELS].choices = layers
    # # selecting layers
    widget_ultrack._cb_images[UltrackInput.LABELS].value = layers['segments']

    #load config
    widget_ultrack._data_forms.load_config(config)


    widget_hier = HierarchyVizWidget(viewer2)
    viewer2.window.add_dock_widget(widget_hier)

    assert "hierarchy" in viewer.layers

    #test moving sliders:
    widget._slider_update(0.75)
    widget._slider_update(0.25)

    # test is shape of layer.data has same shape as the data shape reported in config:
    assert tuple(config.data_config.metadata["shape"]) == viewer2.layers['hierarchy'].data.shape     #metadata["shape"] is a list, data.shape in layer is a tuple

    if request.config.getoption("--show-napari-viewer"):
        napari.run()