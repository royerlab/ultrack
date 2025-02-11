from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest
from napari.viewer import ViewerModel

from ultrack.tracks.video import (
    tracks_df_to_3D_video,
    tracks_df_to_moving_2D_plane_video,
    tracks_df_to_videos,
)
from ultrack.utils.constants import NO_PARENT

pytest.importorskip("napari_animation")


@pytest.fixture
def tracks_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "parent_id": [NO_PARENT, 0, 1, 2, 1, 4],
            "track_id": [1, 1, 2, 2, 3, 3],
            "parent_track_id": [NO_PARENT, NO_PARENT, 1, 1, 1, 1],
            "t": [0, 1, 2, 3, 2, 3],
            "z": [0, 0, 0, 0, 0, 0],
            "y": [10, 20, 30, 10, 20, 10],
            "x": [10, 20, 30, 10, 20, 10],
        }
    )


def test_tracks_video(
    tracks_df: pd.DataFrame,
    make_napari_viewer: Callable[[], ViewerModel],
    tmp_path: Path,
) -> None:

    image = np.random.randint(255, size=(5, 64, 64))

    viewer = make_napari_viewer()
    viewer.add_image(image)

    tracks_df_to_videos(
        viewer,
        tracks_df,
        tmp_path,
    )


def test_tracks_3d_video(
    tracks_df: pd.DataFrame,
    make_napari_viewer: Callable[[], ViewerModel],
    tmp_path: Path,
) -> None:

    image = np.random.randint(255, size=(5, 64, 64, 64))

    viewer = make_napari_viewer()
    im_layer = viewer.add_image(image)

    tracks_df_to_3D_video(
        viewer,
        tracks_df,
        tmp_path / "video.mp4",
        keyframe_step=1,
        update_step=1,
        clipping_planes_layers=[im_layer],
        time_to_camera_kwargs={
            0: {"zoom": 1},
            3: {"zoom": 2, "angles": "auto", "center": "auto"},
        },
    )


def test_tracks_moving_slice_video(
    tracks_df: pd.DataFrame,
    make_napari_viewer: Callable[[], ViewerModel],
    tmp_path: Path,
) -> None:
    image = np.random.randint(255, size=(5, 64, 64))

    viewer = make_napari_viewer()
    viewer.add_image(image)

    tracks_df_to_moving_2D_plane_video(
        viewer,
        tracks_df,
        tmp_path / "2D_slice_video.mp4",
    )
