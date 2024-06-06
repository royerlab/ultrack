import time
from pathlib import Path
from typing import Callable, List

import higra as hg
import napari

from ultrack.config import MainConfig
from ultrack.widgets.ultrackwidget import UltrackWidget


def test_ultrack_widget(
    make_napari_viewer: Callable[[], napari.Viewer],
    zarr_dataset_paths: List[str],
    tmp_path: Path,
) -> None:

    viewer = make_napari_viewer()
    widget = UltrackWidget(viewer)

    layers = viewer.open(zarr_dataset_paths)  # foreground, edge

    # setting combobox choices manually, because they were not working automatically
    widget._main_config_w._foreground_layer_w.choices = layers
    widget._main_config_w._edge_layer_w.choices = layers

    # selecting layers
    widget._main_config_w._foreground_layer_w.value = layers[0]
    widget._main_config_w._edge_layer_w.value = layers[1]

    # checking if widget changes are propagated to children config from multiple interfaces
    config = MainConfig()
    config.segmentation_config.threshold = 0.65

    # temporary working dir
    config.data_config.working_dir = tmp_path

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
    # test save config
    widget._main_config_w._config_loader_w.value = tmp_path / "config.toml"
    widget._main_config_w._save_config_btn.clicked.emit()

    delay = 4  # delay necessary due to thread worker, blocking while loop did not work. I don't know why

    widget._segmentation_w._segment_btn.clicked.emit()
    time.sleep(delay)

    widget._linking_w._link_btn.clicked.emit()
    time.sleep(delay)

    widget._tracking_w._track_btn.clicked.emit()
    time.sleep(delay)

    # checking if the whole thing can be run twice --- overwrite is working
    widget._segmentation_w._segment_btn.clicked.emit()
    time.sleep(delay)

    # checking linking with features edge weight
    widget._linking_w._images_w.value = True
    viewer.layers.selection.select_only(viewer.layers["contours"])

    widget._linking_w._link_btn.clicked.emit()
    time.sleep(delay)

    widget._tracking_w._track_btn.clicked.emit()
    time.sleep(delay)
