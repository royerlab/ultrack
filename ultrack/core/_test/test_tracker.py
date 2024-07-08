from pathlib import Path
from typing import Tuple

import networkx as nx
import numpy as np
import pytest
import torch as th
import torch.nn.functional as F
import zarr

import ultrack
from ultrack import MainConfig, link, segment, solve, to_ctc, to_tracks_layer, track
from ultrack.core.export import tracks_layer_to_networkx, tracks_layer_to_trackmate
from ultrack.core.tracker import TrackerStatus
from ultrack.imgproc.flow import add_flow, identity_grid


@pytest.fixture
def mock_flow_field(request) -> np.ndarray:
    ndim = request.param["ndim"]
    size = request.param["size"]
    length = request.param["length"]
    theta = th.tensor(0.5)
    shape = tuple(size for _ in range(ndim))
    if ndim == 2:
        T = th.tensor(
            [[th.cos(theta), -th.sin(theta), 0], [th.sin(theta), th.cos(theta), 0]]
        )
    else:
        T = th.tensor(
            [
                [th.cos(theta), -th.sin(theta), 0, 0],
                [th.sin(theta), th.cos(theta), 0, 0],
                [0, 0, 1, 0],
            ]
        )
    T = T[None, ...]
    grid_shape = (1, 1) + shape
    mock_grid = F.affine_grid(T, grid_shape, align_corners=True)

    flow = identity_grid(shape) - mock_grid
    flow = flow.movedim(-1, 1)
    flow = th.cat([flow for _ in range(length)], dim=0)
    return flow.numpy() / 2


@pytest.mark.parametrize(
    "config_content,timelapse_mock_data",
    [
        ({"segmentation.n_workers": 4}, {"length": 4, "size": 128, "n_dim": 2}),
        ({"segmentation.n_workers": 4}, {"length": 4, "size": 64, "n_dim": 3}),
    ],
    indirect=True,
)
def test_track(
    config_instance: MainConfig,
    timelapse_mock_data: Tuple[zarr.Array, zarr.Array, zarr.Array],
) -> None:
    foreground, contours, labels = timelapse_mock_data

    # 1st track call using labels
    track(config_instance, labels=labels)
    df_original, graph_original = to_tracks_layer(config_instance)

    tracker = ultrack.Tracker(config_instance)
    tracker.track(labels=labels, overwrite=True)
    df_tracker, graph_tracker = to_tracks_layer(config_instance)

    assert df_original.equals(df_tracker)
    assert graph_original == graph_tracker

    # 2nd track call using foreground and contours
    track(config_instance, foreground=foreground, contours=contours, overwrite=True)
    df_original, graph_original = to_tracks_layer(config_instance)

    tracker = ultrack.Tracker(config_instance)
    tracker.track(foreground=foreground, contours=contours, overwrite=True)
    df_tracker, graph_tracker = to_tracks_layer(config_instance)

    assert df_original.equals(df_tracker)
    assert graph_original == graph_tracker


@pytest.mark.parametrize(
    "config_content,timelapse_mock_data",
    [
        ({"segmentation.n_workers": 4}, {"length": 4, "size": 128, "n_dim": 2}),
        ({"segmentation.n_workers": 4}, {"length": 4, "size": 64, "n_dim": 3}),
    ],
    indirect=True,
)
def test_link_segment_and_solve(
    config_instance: MainConfig,
    timelapse_mock_data: Tuple[zarr.Array, zarr.Array, zarr.Array],
) -> None:
    foreground, contours, _ = timelapse_mock_data

    tracker = ultrack.Tracker(config_instance)
    tracker.segment(foreground=foreground, contours=contours)
    tracker.link()
    tracker.solve()

    df, graph = to_tracks_layer(config_instance)

    segment(
        foreground=foreground, contours=contours, config=config_instance, overwrite=True
    )
    link(config_instance, overwrite=True)
    solve(config_instance, overwrite=True)

    df_original, graph_original = to_tracks_layer(config_instance)

    assert df.equals(df_original)
    assert graph == graph_original


@pytest.mark.parametrize(
    "config_content,timelapse_mock_data",
    [
        ({"segmentation.n_workers": 4}, {"length": 4, "size": 128, "n_dim": 2}),
        ({"segmentation.n_workers": 4}, {"length": 4, "size": 64, "n_dim": 3}),
    ],
    indirect=True,
)
def test_outputs(
    config_instance: MainConfig,
    timelapse_mock_data: Tuple[zarr.Array, zarr.Array, zarr.Array],
    tmp_path: str,
) -> None:
    tmp_path = Path(tmp_path)
    foreground, contours, _ = timelapse_mock_data

    tracker = ultrack.Tracker(config_instance)
    tracker.segment(foreground=foreground, contours=contours)
    tracker.link()
    tracker.solve()

    # test to_tracks_layer (and indirectly to_pandas)
    df_tracker, graph_tracker = tracker.to_tracks_layer()
    df_original, graph_original = to_tracks_layer(config_instance)

    assert df_tracker.equals(df_original)
    assert graph_tracker == graph_original

    # test to_ctc
    tracker.to_ctc(output_dir=tmp_path)
    assert (tmp_path / "res_track.txt").exists()
    with open(tmp_path / "res_track.txt") as f:
        content = f.read()

    to_ctc(output_dir=tmp_path, config=config_instance, overwrite=True)
    assert (tmp_path / "res_track.txt").exists()
    with open(tmp_path / "res_track.txt") as f:
        content_original = f.read()

    assert content == content_original

    # test to_zarr
    z_tracker = tracker.to_zarr()
    z_original = ultrack.tracks_to_zarr(config_instance, tracks_df=df_original)

    assert np.array_equal(np.asarray(z_tracker), np.asarray(z_original))

    # test to_trackmate
    out_trackmate_original = tracks_layer_to_trackmate(df_original)
    out_trackmate_tracker = tracker.to_trackmate()

    assert out_trackmate_original == out_trackmate_tracker

    # test to_networkx
    nx_tracker = tracker.to_networkx()
    nx_original = tracks_layer_to_networkx(df_original)

    assert nx.utils.graphs_equal(nx_tracker, nx_original)


@pytest.mark.parametrize(
    "config_content,timelapse_mock_data,mock_flow_field",
    [
        (
            {"segmentation.n_workers": 4},
            {"length": 4, "size": 128, "n_dim": 2},
            {"size": 32, "ndim": 2, "length": 4},
        ),
        (
            {"segmentation.n_workers": 4},
            {"length": 4, "size": 64, "n_dim": 3},
            {"size": 16, "ndim": 3, "length": 4},
        ),
    ],
    indirect=True,
)
def test_flow(
    config_instance: MainConfig,
    timelapse_mock_data: Tuple[zarr.Array, zarr.Array, zarr.Array],
    mock_flow_field: np.ndarray,
) -> None:
    foreground, contours, _ = timelapse_mock_data

    tracker = ultrack.Tracker(config_instance)
    tracker.segment(foreground=foreground, contours=contours)
    tracker.add_flow(mock_flow_field)
    tracker.link()
    tracker.solve()

    df, graph = to_tracks_layer(config_instance)

    segment(
        foreground=foreground, contours=contours, config=config_instance, overwrite=True
    )
    add_flow(config_instance, mock_flow_field)
    link(config_instance, overwrite=True)
    solve(config_instance, overwrite=True)

    df_original, graph_original = to_tracks_layer(config_instance)

    assert df.equals(df_original)
    assert graph == graph_original


def test_error(
    config_instance: MainConfig,
) -> None:
    tracker = ultrack.Tracker(config_instance)

    try:
        tracker.link()
    except ValueError:
        pass

    assert tracker.status == TrackerStatus.NOT_COMPUTED
