from pathlib import Path
from typing import Callable, Tuple

import napari
import zarr
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ultrack.config import MainConfig
from ultrack.core.database import NodeDB, NodeSegmAnnotation
from ultrack.widgets import NodeAnnotationWidget


def test_node_annotation_widget(
    make_napari_viewer: Callable[[], napari.Viewer],
    tracked_database_mock_data: MainConfig,
    timelapse_mock_data: Tuple[zarr.Array, zarr.Array, zarr.Array],
    tmp_path: Path,
    request,
) -> None:
    # NOTE: Use "--show-napari-viewer" to show viewer, useful when debugging

    config = tracked_database_mock_data
    config.data_config.working_dir = tmp_path

    viewer = make_napari_viewer()

    for arr in timelapse_mock_data:
        viewer.add_image(arr)

    widget = NodeAnnotationWidget(viewer)

    viewer.window.add_dock_widget(widget)

    widget.config = config.data_config
    # there was a bug where _nodes would be empty after setting up the config
    assert len(widget._nodes) > 0

    if request.config.getoption("--show-napari-viewer"):
        napari.run()
        return

    assert widget._mask_layer_name in viewer.layers
    assert widget._next_btn.enabled
    assert not widget._prev_btn.enabled
    assert widget.list_index == 0

    widget._next_btn.clicked.emit()
    assert widget._prev_btn.enabled
    assert widget.list_index == 1

    widget._prev_btn.clicked.emit()
    assert not widget._prev_btn.enabled
    assert widget.list_index == 0

    widget._annot_w.value = NodeSegmAnnotation.OVERSEGMENTED
    widget._confirm_btn.clicked.emit()
    assert widget.list_index == 1

    engine = create_engine(widget.config.database_path)
    with Session(engine) as session:
        table_length = session.query(NodeDB.id).count()

    while widget._next_btn.enabled:
        # a bit slow, but worth it
        widget._next_btn.clicked.emit()

    assert widget.list_index == table_length - 1

    widget._annot_w.value = NodeSegmAnnotation.CORRECT
    widget._confirm_btn.clicked.emit()

    # checking if index won't advance when confirming the last node
    assert widget.list_index == table_length - 1

    with Session(engine) as session:
        query = session.query(NodeDB.id)
        n_correct = query.where(
            NodeDB.segm_annotation == NodeSegmAnnotation.CORRECT
        ).count()
        n_oversegmented = query.where(
            NodeDB.segm_annotation == NodeSegmAnnotation.OVERSEGMENTED
        ).count()
        n_undersegmented = query.where(
            NodeDB.segm_annotation == NodeSegmAnnotation.UNDERSEGMENTED
        ).count()

    assert n_correct == 1
    assert n_oversegmented == 1
    assert n_undersegmented == 0
