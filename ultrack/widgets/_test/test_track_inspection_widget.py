from pathlib import Path
from typing import Callable

import napari
import numpy as np
import pytest

from ultrack import to_tracks_layer
from ultrack.config import MainConfig
from ultrack.widgets import TrackInspectionWidget


@pytest.mark.parametrize(
    "config_content,timelapse_mock_data",
    [
        (
            {
                "segmentation.n_workers": 4,
                "linking.n_workers": 4,
                "tracking.appear_weight": -0.25,
                "tracking.disappear_weight": -0.5,
                "tracking.division_weight": -0.25,
            },
            {"length": 4},
        )
    ],
    indirect=True,
)
def test_division_annotation_widget(
    make_napari_viewer: Callable[[], napari.Viewer],
    tracked_database_mock_data: MainConfig,
    tmp_path: Path,
    request,
) -> None:
    # NOTE: Use "--show-napari-viewer" to show viewer, useful when debugging

    config = tracked_database_mock_data
    config.data_config.working_dir = tmp_path

    viewer = make_napari_viewer()
    widget = TrackInspectionWidget(viewer)
    viewer.window.add_dock_widget(widget)

    assert not widget._next_btn.enabled
    assert not widget._prev_btn.enabled

    tracks, graph = to_tracks_layer(config.data_config)
    tracks = tracks.to_numpy()
    tracks[:, 2:] += np.random.uniform(3, size=tracks[:, 2:].shape)

    tracks_layer = viewer.add_tracks(tracks, graph=graph)

    if request.config.getoption("--show-napari-viewer"):
        napari.run()
        return

    assert widget._tracks_layer_w.value == tracks_layer
    assert len(widget._sorted_tracks) > 0

    next_count = 0
    while widget._next_btn.enabled:
        widget._next_btn.clicked()
        next_count += 1
        assert widget._current_track_layer is not None

    prev_count = 0
    while widget._prev_btn.enabled:
        widget._prev_btn.clicked()
        prev_count += 1
        assert widget._current_track_layer is not None

    assert next_count > 0
    assert prev_count == next_count
