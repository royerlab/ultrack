import pickle
from typing import Tuple

import pandas as pd
import pytest
import zarr

from ultrack.config.config import MainConfig
from ultrack.core.segmentation.dbbase import NodeDB, get_database_path
from ultrack.core.segmentation.processing import segment


@pytest.mark.parametrize(
    "config_content,timelapse_mock_data",
    [
        ({"segmentation.n_workers": 1}, {"length": 4, "size": 128, "n_dim": 2}),
        ({"segmentation.n_workers": 4}, {"length": 4, "size": 128, "n_dim": 2}),
        ({"segmentation.n_workers": 1}, {"length": 4, "size": 64, "n_dim": 3}),
        ({"segmentation.n_workers": 4}, {"length": 4, "size": 64, "n_dim": 3}),
    ],
    indirect=True,
)
def test_multiprocessing(
    config_instance: MainConfig,
    timelapse_mock_data: Tuple[zarr.Array, zarr.Array],
) -> None:
    detection, edges = timelapse_mock_data

    segment(
        detection,
        edges,
        config_instance.segmentation_config,
        config_instance.working_dir,
    )

    db_path = get_database_path(config_instance.working_dir, "sqlite")
    df = pd.read_sql_table(NodeDB.__tablename__, con=db_path)

    # assert all columns are present
    assert set(NodeDB.__table__.columns.keys()) == set(df.columns)

    # assert unpickling works
    for blob in df["pickle"]:
        pickle.loads(blob)
