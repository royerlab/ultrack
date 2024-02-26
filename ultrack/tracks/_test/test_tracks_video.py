from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest
from napari.viewer import ViewerModel

from ultrack.core.database import NO_PARENT
from ultrack.tracks.video import tracks_df_to_videos

pytest.importorskip("napari_animation")


def test_tracks_video(
    make_napari_viewer: Callable[[], ViewerModel],
    tmp_path: Path,
) -> None:

    tracks_df = pd.DataFrame(
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

    image = np.random.randint(255, size=(5, 64, 64))

    viewer = make_napari_viewer()
    viewer.add_image(image)

    tracks_df_to_videos(
        viewer,
        tracks_df,
        tmp_path,
    )
