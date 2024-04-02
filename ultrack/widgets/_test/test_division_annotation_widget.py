from pathlib import Path
from typing import Callable, Tuple

import napari
import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ultrack.config import MainConfig
from ultrack.core.database import NodeDB, VarAnnotation
from ultrack.widgets import DivisionAnnotationWidget


@pytest.mark.parametrize(
    "config_content",
    [
        {
            "segmentation.min_area": 25,
            "tracking.division_weight": 0,
        }
    ],
    indirect=True,
)
def test_division_annotation_widget(
    make_napari_viewer: Callable[[], napari.Viewer],
    cell_division_mock_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    tracked_cell_division_mock_data: MainConfig,
    tmp_path: Path,
    request,
) -> None:
    # NOTE: Use "--show-napari-viewer" to show viewer, useful when debugging

    config = tracked_cell_division_mock_data
    config.data_config.working_dir = tmp_path

    viewer = make_napari_viewer()

    for arr in cell_division_mock_data:
        viewer.add_image(arr)

    widget = DivisionAnnotationWidget(viewer)

    viewer.window.add_dock_widget(widget)

    widget.config = config.data_config
    assert len(widget._nodes) > 0

    if request.config.getoption("--show-napari-viewer"):
        napari.run()
        return

    assert widget.list_index == 0

    widget._annot_w.value = VarAnnotation.REAL
    widget._confirm_btn.clicked.emit()
    assert widget.list_index == 1

    widget._annot_w.value = VarAnnotation.FAKE
    widget._confirm_btn.clicked.emit()
    assert widget.list_index == 1  # there are only two divisions

    engine = create_engine(widget.config.database_path)
    with Session(engine) as session:
        div_annot = [
            annot
            for annot, in session.query(NodeDB.division).where(
                NodeDB.division != VarAnnotation.UNKNOWN
            )
        ]

    assert len(div_annot) == 2
    assert VarAnnotation.REAL in div_annot
    assert VarAnnotation.FAKE in div_annot
