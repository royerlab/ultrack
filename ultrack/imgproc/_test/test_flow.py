import platform

import numpy as np
import pytest
import sqlalchemy as sqla
from sqlalchemy.orm import Session

from ultrack.config import MainConfig
from ultrack.core.database import Base, NodeDB
from ultrack.imgproc.flow import (
    add_flow,
    advenct_field,
    timelapse_flow,
    trajectories_to_tracks,
)


@pytest.mark.skipif(
    platform.machine() == "arm64", reason="Test skipped on Apple Silicon"
)
@pytest.mark.parametrize("ndim", [2, 3])
def test_flow_field(ndim: int, request) -> None:
    th = pytest.importorskip("torch")

    size = (64,) * ndim
    sigma = 15
    im_factor = 2
    grid_factor = 4

    grid = th.stack(th.meshgrid([th.arange(s) for s in size], indexing="ij"), dim=-1)

    mus = th.Tensor(
        [[0.5, 0.5, 0.5], [0.55, 0.5, 0.5], [0.57, 0.48, 0.53], [0.55, 0.45, 0.55]]
    )[:, :ndim]

    mus = (mus * th.tensor(size)).round().int()

    frames = th.stack([th.exp(-th.square(grid - mu).sum(dim=-1) / sigma) for mu in mus])

    fields = timelapse_flow(
        frames.cpu().numpy(),
        im_factor=im_factor,
        grid_factor=grid_factor,
        num_iterations=2_000,
        lr=0.0001,
    )
    trajectory = advenct_field(fields, mus[None, 0], size)
    tracks = trajectories_to_tracks(trajectory)

    if request.config.getoption("--show-napari-viewer"):
        import napari

        kwargs = {"blending": "additive", "interpolation3d": "nearest", "rgb": False}

        viewer = napari.Viewer()

        viewer.add_image(frames.cpu().numpy(), **kwargs)
        viewer.add_tracks(tracks)
        viewer.add_image(
            fields,
            colormap="turbo",
            scale=(im_factor,) * ndim,
            channel_axis=1,
            **kwargs,
        )

        napari.run()

    # tolerance is super loose, should be improved, but algorithm is not precise
    th.testing.assert_close(trajectory.squeeze(), mus.half(), atol=0.0, rtol=0.05)


@pytest.mark.parametrize("n_channels,n_dim", [(2, 2), (3, 3), (2, 3)])
def test_add_flow(
    config_instance: MainConfig,
    n_channels: int,
    n_dim: int,
) -> None:
    # non-square shape
    whole_shape = (213, 157, 256)
    shape = whole_shape[-n_dim:]
    n_time_pts = 2

    data_config = config_instance.data_config

    engine = sqla.create_engine(data_config.database_path)
    Base.metadata.create_all(engine)
    data_config.metadata_add({"shape": (n_time_pts,) + shape})

    time = [1, 1, 1]
    coordinates = np.asarray(
        [
            [100, 100, 100],  # (0, 0, 0)
            [40, 140, 255],  # (0, 1, 1)
            [172, 56, 200],  # (1, 0, 1)
        ]
    )
    coordinates[:, :-n_dim] = 0.0

    vec_coordinates = coordinates.astype(float) / whole_shape
    vec_coordinates = np.round(vec_coordinates).astype(int)
    # vec_coordinates = [
    #     (0, 0, 0),
    #     (0, 1, 1),
    #     (1, 0, 1),
    # ]

    vector_field = np.arange(2**n_dim).reshape((2,) * n_dim)
    vector_field = np.tile(vector_field, (n_time_pts,) + (1,) * n_dim)
    vector_field = [vector_field * i for i in range(1, n_channels + 1)]
    """
    2D vector field
    y:
    [[0, 1],
     [2, 3]]
    x:
    [[0, 2],
     [4, 6]]

    3D vector field:

    z (y):
    [[0, 1], [[4, 5],
     [2, 3]]  [6, 7]]

    y (x):
    [[0, 2], [[8, 10],
     [4, 6]]  [12, 14]]

    x (optional):
    [[0, 3], [[12, 15],
     [6, 9]]  [18, 21]]
    """

    # normalizing vector field
    vector_field = [v / s for v, s in zip(vector_field, shape[-n_channels:])]

    with Session(engine) as session:
        for i, (t, centroids) in enumerate(zip(time, coordinates)):
            mock_node = NodeDB(
                id=i + 1,
                t_node_id=-1,
                t_hier_id=-1,
                area=0.0,
                t=t,
                z=centroids[0],
                y=centroids[1],
                x=centroids[2],
                pickle=b"0000",
            )
            session.add(mock_node)
        session.commit()

    add_flow(config_instance, vector_field)

    with Session(engine) as session:
        query = session.query(
            NodeDB.id,
            NodeDB.z_shift,
            NodeDB.y_shift,
            NodeDB.x_shift,
        )
        results = {i: (z, y, x) for i, z, y, x in query}

    for i, c in enumerate(vec_coordinates):
        index = (time[i],) + tuple(c[-n_dim:])
        expected_shift = tuple(
            v[index] * s for v, s in zip(vector_field, shape[-n_channels:])
        )
        expected_shift = (0,) * (3 - len(expected_shift)) + expected_shift
        assert results[i + 1] == expected_shift
