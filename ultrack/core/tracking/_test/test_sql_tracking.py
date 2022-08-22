import numpy as np
import pandas as pd
import pytest

from ultrack import track
from ultrack.config.config import MainConfig
from ultrack.core.database import NO_PARENT, NodeDB

_TEST_PARAMS = (
    {
        "segmentation.n_workers": 4,
        "linking.n_workers": 4,
        "tracking.appear_weight": -0.25,
        "tracking.disappear_weight": -0.5,
        "tracking.division_weight": -0.25,
        "tracking.window_size": 5,
        "tracking.overlap_size": 2,
    },
    {
        "length": 10,
    },
)


def _validate_tracking_solution(config: MainConfig):
    nodes = pd.read_sql_query(
        f"SELECT id, t, selected, parent_id FROM {NodeDB.__tablename__}",
        con=config.data_config.database_path,
        index_col="id",
    )

    # at least one node was selected
    assert np.any(nodes["selected"])

    # assert parents id are valid
    assert np.all(
        nodes.loc[nodes["parent_id"] != NO_PARENT, "parent_id"].isin(nodes.index)
    )

    # assert starting nodes are parentless
    assert np.all(nodes.loc[nodes["t"] == 0, "parent_id"] == NO_PARENT)

    # assert there isn't any disconnected chunk
    for t, group in nodes.groupby("t"):
        if t == 0:
            continue
        assert np.any(group["parent_id"] != NO_PARENT)


@pytest.mark.parametrize(
    "config_content,timelapse_mock_data",
    [_TEST_PARAMS],
    indirect=True,
)
def test_sql_tracking(
    linking_database_mock_data: MainConfig,
) -> None:
    config = linking_database_mock_data

    track(config.tracking_config, config.data_config)

    _validate_tracking_solution(config)


@pytest.mark.parametrize(
    "config_content,timelapse_mock_data",
    [_TEST_PARAMS],
    indirect=True,
)
def test_batch_sql_tracking(
    linking_database_mock_data: MainConfig,
) -> None:
    config = linking_database_mock_data

    track(config.tracking_config, config.data_config, indices=0)
    track(config.tracking_config, config.data_config, indices=1)

    with pytest.raises(ValueError):
        track(config.tracking_config, config.data_config, indices=2)

    _validate_tracking_solution(config)
