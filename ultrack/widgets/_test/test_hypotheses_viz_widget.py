from pathlib import Path
from typing import Callable, Dict

import napari
import numpy as np
import pytest
import sqlalchemy as sqla
from sqlalchemy.orm import Session

from ultrack.config import MainConfig
from ultrack.core.database import NodeDB
from ultrack.core.segmentation.node import Node
from ultrack.widgets import HypothesesVizWidget


def _is_sorted(nodes: Dict[int, Node]) -> bool:
    nodes = list(nodes.values())
    return all(nodes[i].area <= nodes[i + 1].area for i in range(len(nodes) - 1))


def test_hypotheses_viz_widget(
    make_napari_viewer: Callable[[], napari.Viewer],
    linked_database_mock_data: MainConfig,
    tmp_path: Path,
    request,
) -> None:
    # NOTE: Use "--show-napari-viewer" to show viewer, useful when debugging

    config = linked_database_mock_data
    config.data_config.working_dir = tmp_path

    viewer = make_napari_viewer()

    widget = HypothesesVizWidget(viewer)
    widget.config = config.data_config
    viewer.window.add_dock_widget(widget)

    with pytest.warns(UserWarning):
        assert widget._time == 0

    with pytest.warns(UserWarning):
        widget._load_btn.clicked.emit()

    assert _is_sorted(widget._nodes)

    # forcing every node to be drawn
    widget._area_threshold_w.value = widget._area_threshold_w.max

    labels = viewer.layers[widget._segm_layer_name].data

    engine = sqla.create_engine(config.data_config.database_path)
    with Session(engine) as session:
        query = session.query(NodeDB.pickle).where(NodeDB.t == 0)

    # checking if every node was drawn
    for (node,) in query:
        assert np.all(labels[node.mask_indices()])

    widget._load_neighbors(0)
    assert widget._link_layer_name not in viewer.layers

    # testing using last `node`
    widget._load_neighbors(node.id)
    assert widget._link_layer_name in viewer.layers

    if request.config.getoption("--show-napari-viewer"):
        napari.run()
