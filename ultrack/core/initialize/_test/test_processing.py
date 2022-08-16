import pickle
from pathlib import Path
from typing import Tuple

import pandas as pd
import pytest
import zarr

from ultrack.config.config import MainConfig
from ultrack.core.initialize.dbbase import NodeDB, get_database_path
from ultrack.core.initialize.processing import add_nodes_to_database


@pytest.mark.parametrize(
    "config_content,timelapse_mock_data",
    [
        ({"init.n_workers": 1}, {"length": 4, "size": 128, "n_dim": 2}),
        ({"init.n_workers": 4}, {"length": 4, "size": 128, "n_dim": 2}),
        ({"init.n_workers": 1}, {"length": 4, "size": 64, "n_dim": 3}),
        ({"init.n_workers": 4}, {"length": 4, "size": 64, "n_dim": 3}),
    ],
    indirect=True,
)
def test_processing(
    config_instance: MainConfig,
    timelapse_mock_data: Tuple[zarr.Array, zarr.Array],
    tmp_path: Path,
) -> None:
    detection, edges = timelapse_mock_data

    init_config = config_instance.init_config

    add_nodes_to_database(detection, edges, init_config, tmp_path)

    db_path = get_database_path(tmp_path, "sqlite")
    df = pd.read_sql_table(NodeDB.__tablename__, con=db_path)

    # assert all columns are present
    assert set(NodeDB.__table__.columns.keys()) == set(df.columns)

    # assert unpickling works
    for blob in df["pickle"]:
        pickle.loads(blob)
