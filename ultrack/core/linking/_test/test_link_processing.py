from types import SimpleNamespace
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import zarr

from ultrack import link
from ultrack.config import MainConfig
from ultrack.config.config import LinkingConfig
from ultrack.core.database import LinkDB, NodeDB
from ultrack.core.linking.processing import compute_spatial_neighbors


@pytest.mark.parametrize(
    "config_content",
    [
        {"linking.n_workers": 1, "linking.max_distance": 1000},
        {
            "linking.n_workers": 4,
            "linking.max_distance": 1000,
            "linking.max_neighbors": 5,
        },
        {"linking.n_workers": 4, "linking.max_distance": 10},
    ],
    indirect=True,
)
def test_multiprocess_link(
    segmentation_database_mock_data: MainConfig,
    timelapse_mock_data: Tuple[zarr.Array, zarr.Array, zarr.Array],
) -> None:
    config = segmentation_database_mock_data

    link(
        config,
        scale=(2, 1, 1),
        images=[
            timelapse_mock_data[0]
        ],  # using a random image just to test with color verify
    )

    edges = pd.read_sql_table(
        LinkDB.__tablename__, con=config.data_config.database_path
    )

    # since they're all the same, there must be at one edge with weight 1.0 for each node
    for _, group in edges.groupby("target_id"):
        assert len(group) <= config.linking_config.max_neighbors
        assert (group["weight"] == 1.0).sum() == 1.0

    # validate if distances are whitin max_distance
    nodes = pd.read_sql_query(
        f"SELECT id, t, z, y, x FROM {NodeDB.__tablename__}",
        con=config.data_config.database_path,
        index_col="id",
    )

    source_pos = nodes.loc[edges["source_id"], ["z", "y", "x"]].values
    target_pos = nodes.loc[edges["target_id"], ["z", "y", "x"]].values

    distances = np.linalg.norm(target_pos - source_pos, axis=1)
    assert np.all(distances < config.linking_config.max_distance)

    # assert every node is present in edges --- not necessary in the real case, but here we know it's true.
    last_t = nodes["t"].max()
    assert np.all(nodes[nodes["t"] != last_t].index.isin(edges["source_id"]))
    assert np.all(nodes[nodes["t"] != 0].index.isin(edges["target_id"]))


def _stub_node(z: int, y: int, x: int) -> SimpleNamespace:
    """Lightweight Node stand-in exposing just the `centroid` attribute.

    `compute_spatial_neighbors` only reads `n.centroid` on the path exercised
    here (it returns early before any other Node API is touched), so a real
    `Node` — which requires a parent and database setup — is not needed.
    """
    return SimpleNamespace(centroid=np.asarray([z, y, x], dtype=np.float32))


@pytest.mark.parametrize(
    "empty_side, expected_msg",
    [
        ("source", "No segments found at source t=3"),
        ("target", "No segments found at target t=4"),
        ("both", "No segments found at source t=3"),
    ],
)
def test_compute_spatial_neighbors_empty_frame_warns(
    empty_side: str, expected_msg: str, tmp_path
) -> None:
    """Regression test for issue #274.

    `compute_spatial_neighbors` used to raise `IndexError: tuple index out of
    range` when either source or target nodes were empty, because
    `np.asarray([]).shape[1]` is undefined. It should now warn (pointing at the
    offending t-frame) and skip linking instead of crashing.
    """
    nodes = [_stub_node(0, 0, 0), _stub_node(1, 1, 1)]
    source_nodes = [] if empty_side in ("source", "both") else nodes
    target_nodes = [] if empty_side in ("target", "both") else nodes
    target_shift = np.zeros((len(target_nodes), 3), dtype=np.float32)

    with pytest.warns(RuntimeWarning, match=expected_msg):
        compute_spatial_neighbors(
            time=3,
            config=LinkingConfig(),
            source_nodes=source_nodes,
            target_nodes=target_nodes,
            target_shift=target_shift,
            scale=None,
            table_name=LinkDB.__tablename__,
            db_path=f"sqlite:///{tmp_path / 'unused.db'}",
            images=[],
        )
