from typing import Callable, List

import napari
import numpy as np
import pytest
import sqlalchemy as sqla
from sqlalchemy.orm import Session

from ultrack.config import MainConfig
from ultrack.core.database import NodeDB
from ultrack.core.segmentation.node import Node
from ultrack.widgets.segm_vis_widget import SegmVizWidget


def _is_sorted(nodes: List[Node]) -> bool:
    return all(nodes[i].area <= nodes[i + 1].area for i in range(len(nodes) - 1))


def test_segm_vis_widget(
    make_napari_viewer: Callable[[], napari.Viewer],
    segmentation_database_mock_data: MainConfig,
    request,
) -> None:
    # NOTE: Use "--show-napari-viewer" to show viewer, useful when debugging

    config = segmentation_database_mock_data

    viewer = make_napari_viewer()

    widget = SegmVizWidget(viewer)
    widget.config = config.data_config
    viewer.window.add_dock_widget(widget)

    with pytest.warns(UserWarning):
        assert widget._time == 0

    with pytest.warns(UserWarning):
        widget._load_btn.clicked.emit()

    assert _is_sorted(widget._nodes)

    # forcing every node to be drawn
    widget._area_threshold_w.value = widget._area_threshold_w.max

    labels = viewer.layers[widget._layer_name].data

    engine = sqla.create_engine(config.data_config.database_path)
    with Session(engine) as session:
        query = session.query(NodeDB.pickle).where(NodeDB.t == 0)

    # checking if every node was drawn
    for (node,) in query:
        assert np.all(labels[node.mask_indices()])

    if request.config.getoption("--show-napari-viewer"):
        napari.run()
