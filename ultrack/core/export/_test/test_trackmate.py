from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ultrack.core.export.trackmate import tracks_layer_to_trackmate
from ultrack.utils.constants import NO_PARENT

pytrackmate = pytest.importorskip("pytrackmate")


def test_trackmate_export_spot_match(tmp_path: Path) -> None:
    """Check if the spots (objects) match between the tracks and the exported trackmate xml file.

    This test cannot check if the exported tracking is valid.
    """
    tracks_outpath = tmp_path / "tracks.xml"

    tracks_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "parent_id": [NO_PARENT, 1, 2, 2],
            "track_id": [1, 1, 2, 3],
            "t": [0, 1, 2, 2],
            "z": [0, 0, 0, 0],
            "y": [10, 20, 30, 10],
            "x": [1, 2, 3, 1],
        }
    )

    xml_str = tracks_layer_to_trackmate(tracks_df)
    with open(tracks_outpath, "w") as f:
        f.write(xml_str)

    trackmate_df = pytrackmate.trackmate_peak_import(tracks_outpath)
    print(trackmate_df)

    assert trackmate_df.shape[0] == tracks_df.shape[0]

    np.testing.assert_allclose(
        tracks_df[["t", "z", "y", "x"]], trackmate_df[["t_stamp", "z", "y", "x"]]
    )
