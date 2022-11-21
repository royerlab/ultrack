import numpy as np
import pandas as pd
import pytest

from ultrack import link
from ultrack.config import MainConfig
from ultrack.core.database import LinkDB, NodeDB


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
) -> None:
    config = segmentation_database_mock_data

    link(config.linking_config, config.data_config)

    edges = pd.read_sql_table(
        LinkDB.__tablename__, con=config.data_config.database_path
    )

    # since they're all the same, there must be at one edge with weight 1.0 for each node
    for _, group in edges.groupby("target_id"):
        assert len(group) <= config.linking_config.max_neighbors
        assert (group["iou"] == 1.0).sum() == 1.0

    # validate if distances are whitin max_distance
    nodes = pd.read_sql_query(
        f"SELECT id, t, z, y, x FROM {NodeDB.__tablename__}",
        con=config.data_config.database_path,
        index_col="id",
    )

    source_pos = nodes.loc[edges["source_id"], ["z", "y", "x"]].values
    target_pos = nodes.loc[edges["target_id"], ["z", "y", "x"]].values

    distances = np.linalg.norm(target_pos - source_pos)
    assert np.all(distances < config.linking_config.max_distance)

    # assert every node is present in edges --- not necessary in the real case, but here we know it's true.
    last_t = nodes["t"].max()
    assert np.all(nodes[nodes["t"] != last_t].index.isin(edges["source_id"]))
    assert np.all(nodes[nodes["t"] != 0].index.isin(edges["target_id"]))
