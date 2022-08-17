from typing import Callable

import higra as hg
import napari

from ultrack.config import MainConfig
from ultrack.widgets.ultrackwidget import UltrackWidget


def test_ultrack_widget(make_napari_viewer: Callable[[], napari.Viewer]) -> None:
    viewer = make_napari_viewer()
    widget = UltrackWidget(viewer)

    # checking if widget changes are propagated to children config from multiple interfaces
    config = MainConfig()
    config.segmentation_config.threshold = 0.65

    widget.config = config
    assert widget._segmentation_w._attr_to_widget["threshold"].value == 0.65
    assert widget._segmentation_w.config.threshold == 0.65
    assert widget.config.segmentation_config.threshold == 0.65

    widget._segmentation_w._attr_to_widget["threshold"].value = 0.42
    assert widget.config.segmentation_config.threshold == 0.42
    assert widget._segmentation_w.config.threshold == 0.42

    # checking if func combo is working
    widget._segmentation_w._attr_to_widget[
        "ws_hierarchy"
    ].value = hg.watershed_hierarchy_by_dynamics
    assert (
        widget.config.segmentation_config.ws_hierarchy
        == hg.watershed_hierarchy_by_dynamics
    )
