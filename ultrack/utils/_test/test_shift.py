import numpy as np
import pytest
import sqlalchemy as sqla
from sqlalchemy.orm import Session

from ultrack.config import MainConfig
from ultrack.core.database import Base, NodeDB
from ultrack.utils.shift import add_shift


@pytest.mark.parametrize("n_channels,n_dim", [(2, 2), (3, 3), (2, 3)])
def test_add_shift(
    config_instance: MainConfig,
    n_channels: int,
    n_dim: int,
) -> None:
    # simulating a (256,) 256, 256 dataset
    shape = (256, 256, 256)[-n_dim:]
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

    vec_coordinates = [
        (0, 0, 0),
        (0, 1, 1),
        (1, 0, 1),
    ]

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

    add_shift(data_config, vector_field)

    with Session(engine) as session:
        query = session.query(
            NodeDB.id,
            NodeDB.z_shift,
            NodeDB.y_shift,
            NodeDB.x_shift,
        )
        results = {i: (z, y, x) for i, z, y, x in query}

    for i, c in enumerate(vec_coordinates):
        index = (time[i],) + c[-n_dim:]
        expected_shift = tuple(v[index] for v in vector_field)
        expected_shift = (0,) * (3 - len(expected_shift)) + expected_shift
        assert results[i + 1] == expected_shift
