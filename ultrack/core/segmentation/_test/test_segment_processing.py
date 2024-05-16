import pickle
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import zarr

from ultrack import segment
from ultrack.config.config import MainConfig
from ultrack.core.database import NodeDB, OverlapDB


@pytest.mark.parametrize(
    "config_content,timelapse_mock_data",
    [
        (
            {"segmentation.n_workers": 1, "segmentation.max_noise": 0.1},
            {"length": 4, "size": 128, "n_dim": 2},
        ),
        (
            {"segmentation.n_workers": 4, "segmentation.max_noise": 0.1},
            {"length": 4, "size": 128, "n_dim": 2},
        ),
        (
            {"segmentation.n_workers": 1, "segmentation.max_noise": 0.1},
            {"length": 4, "size": 64, "n_dim": 3},
        ),
        (
            {"segmentation.n_workers": 4, "segmentation.max_noise": 0.1},
            {"length": 4, "size": 64, "n_dim": 3},
        ),
    ],
    indirect=True,
)
def test_multiprocessing_segment(
    config_instance: MainConfig,
    timelapse_mock_data: Tuple[zarr.Array, zarr.Array, zarr.Array],
) -> None:
    foreground, contours, _ = timelapse_mock_data

    segment(
        foreground,
        contours,
        config_instance,
    )

    assert config_instance.data_config.metadata["shape"] == list(contours.shape)

    df = pd.read_sql_table(
        NodeDB.__tablename__, con=config_instance.data_config.database_path
    )

    # assert all columns are present
    assert set(NodeDB.__table__.columns.keys()) == set(df.columns)

    nodes = {}
    # assert unpickling works
    for blob in df["pickle"]:
        node = pickle.loads(blob)
        nodes[node.id] = node

    overlaps = pd.read_sql_table(
        OverlapDB.__tablename__, con=config_instance.data_config.database_path
    )

    assert np.all(overlaps["node_id"] != overlaps["ancestor_id"])

    # asserting they really overlap
    for node_id, ancestor_id in zip(overlaps["node_id"], overlaps["ancestor_id"]):
        assert nodes[node_id].IoU(nodes[ancestor_id]) > 0.0

    overlaps = set(zip(overlaps["node_id"].tolist(), overlaps["ancestor_id"].tolist()))

    # assert we didn't miss any overlap
    for _, group in df.groupby(["t", "t_hier_id"]):
        indices = group["id"].tolist()
        for i in indices:
            node_i = nodes[i]
            for j in indices:
                if i == j or (i, j) in overlaps or (j, i) in overlaps:
                    # overlaps have alreayd been checked
                    continue
                node_j = nodes[j]
                assert node_i.IoU(node_j) == 0.0
