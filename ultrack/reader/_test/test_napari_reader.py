from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import pytest
from napari.plugins import _initialize_plugins
from napari.viewer import ViewerModel

from ultrack.reader.napari_reader import napari_get_reader
from ultrack.utils.constants import NO_PARENT


@pytest.fixture
def tracks_df(n_nodes: int = 10) -> pd.DataFrame:
    coordinates = np.random.rand(n_nodes, 3)
    tracks_data = np.zeros((n_nodes, 5))
    tracks_data[:, 2:] = coordinates

    tracks_data[: n_nodes // 2, 0] = 1
    tracks_data[n_nodes // 2 :, 0] = 2
    tracks_data[: n_nodes // 2, 1] = np.arange(n_nodes // 2)
    tracks_data[n_nodes // 2 :, 1] = np.arange(n_nodes - n_nodes // 2)

    return pd.DataFrame(tracks_data, columns=["track_id", "t", "z", "y", "x"])


def test_reader(tracks_df: pd.DataFrame, tmp_path: Path):
    reader = napari_get_reader("tracks.csv")
    assert reader is None

    path = tmp_path / "good_tracks.csv"
    tracks_df["node_id"] = np.arange(len(tracks_df)) + 1
    tracks_df["labels"] = np.random.randint(2, size=len(tracks_df))
    tracks_df.to_csv(path, index=False)

    reader = napari_get_reader(path)
    assert callable(reader)

    data, kwargs, type = reader(path)[0]
    assert type == "tracks"

    props = kwargs["features"]

    assert np.allclose(props["node_id"], tracks_df["node_id"])
    assert np.allclose(props["labels"], tracks_df["labels"])
    assert np.allclose(data, tracks_df[["track_id", "t", "z", "y", "x"]])


def test_reader_2d(tracks_df: pd.DataFrame, tmp_path: Path):
    reader = napari_get_reader("tracks.csv")
    assert reader is None

    path = tmp_path / "good_tracks.csv"
    tracks_df = tracks_df.drop(columns=["z"])
    tracks_df.to_csv(path, index=False)

    reader = napari_get_reader(path)
    assert callable(reader)

    data, _, type = reader(path)[0]
    assert type == "tracks"

    assert np.allclose(data, tracks_df[["track_id", "t", "y", "x"]])


def test_reader_with_lineage(tmp_path: Path):
    tracks_df = pd.DataFrame(
        {
            "track_id": [1, 1, 2, 3],
            "parent_track_id": [NO_PARENT, NO_PARENT, 1, 1],
            "t": [0, 1, 2, 2],
            "z": [0, 0, 0, 0],
            "y": [10, 20, 30, 10],
            "x": [1, 2, 3, 1],
        }
    )

    path = tmp_path / "tracks.csv"
    tracks_df.to_csv(path, index=False)

    reader = napari_get_reader(path)
    assert callable(reader)

    data, kwargs, type = reader(path)[0]
    assert type == "tracks"
    assert "graph" in kwargs
    assert kwargs["graph"] == {2: 1, 3: 1}

    assert np.allclose(data, tracks_df[["track_id", "t", "z", "y", "x"]])


def test_non_existing_track():
    reader = napari_get_reader("tracks.csv")
    assert reader is None


def test_wrong_columns_track(tracks_df: pd.DataFrame, tmp_path: Path):
    reader = napari_get_reader("tracks.csv")
    assert reader is None

    path = tmp_path / "bad_tracks.csv"
    tracks_df = tracks_df.rename(columns={"track_id": "id"})
    tracks_df.to_csv(path, index=False)
    reader = napari_get_reader(path)
    assert reader is None


def test_napari_viewer_open_tracks(
    make_napari_viewer: Callable[[], ViewerModel],
    tracks_df: pd.DataFrame,
    tmp_path: Path,
) -> None:

    _initialize_plugins()

    tracks_df.to_csv(tmp_path / "tracks.csv", index=False)

    viewer = make_napari_viewer()
    viewer.open(tmp_path / "tracks.csv", plugin="ultrack")
