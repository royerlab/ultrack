import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sqla
from sqlalchemy.orm import Session

from ultrack import solve, to_tracks_layer
from ultrack.config.config import MainConfig
from ultrack.core.database import LinkDB, NodeDB, VarAnnotation
from ultrack.core.solve.sqltracking import SQLTracking
from ultrack.utils.constants import NO_PARENT

_CONFIG_PARAMS = {
    "segmentation.n_workers": 4,
    "linking.n_workers": 4,
    "tracking.appear_weight": -0.25,
    "tracking.disappear_weight": -0.5,
    "tracking.division_weight": -0.25,
    "tracking.window_size": 5,
    "tracking.overlap_size": 2,
}

_TEST_PARAMS = [
    ({"data.database": "sqlite", **_CONFIG_PARAMS}, {"length": 10}),
    ({"data.database": "postgresql", **_CONFIG_PARAMS}, {"length": 10}),
]


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
    _TEST_PARAMS,
    indirect=True,
)
def test_sql_tracking(
    linked_database_mock_data: MainConfig,
) -> None:
    config = linked_database_mock_data
    solve(config)
    _validate_tracking_solution(config)


@pytest.mark.parametrize(
    "config_content,timelapse_mock_data",
    _TEST_PARAMS,
    indirect=True,
)
def test_batch_sql_tracking(
    linked_database_mock_data: MainConfig,
) -> None:
    config = linked_database_mock_data

    solve(config, batch_index=0)
    solve(config, batch_index=1)

    with pytest.raises(ValueError):
        solve(config, batch_index=2)

    _validate_tracking_solution(config)


@pytest.mark.parametrize(
    "config_content",
    [
        {
            "data.database": "sqlite",
            "segmentation.n_workers": 4,
            "linking.n_workers": 4,
        },
        {
            "data.database": "postgresql",
            "segmentation.n_workers": 4,
            "linking.n_workers": 4,
        },
    ],
    indirect=True,
)
def test_clear_solution(
    tracked_database_mock_data: MainConfig,
) -> None:
    database_path = tracked_database_mock_data.data_config.database_path

    SQLTracking.clear_solution_from_database(database_path)

    engine = sqla.create_engine(database_path)
    with Session(engine) as session:
        assert session.query(LinkDB).count() > 0
        assert session.query(NodeDB).count() > 0
        assert session.query(NodeDB).where(NodeDB.selected).count() == 0
        assert session.query(NodeDB).where(NodeDB.parent_id != NO_PARENT).count() == 0


@pytest.mark.parametrize(
    "config_content,timelapse_mock_data",
    _TEST_PARAMS,
    indirect=True,
)
def test_annotations_sql_tracking(
    linked_database_mock_data: MainConfig,
) -> None:
    config = linked_database_mock_data

    solve(config, overwrite=True, use_annotations=True)
    tracks_df, _ = to_tracks_layer(config)

    engine = sqla.create_engine(config.data_config.database_path)
    with Session(engine) as session:
        session.query(NodeDB).where(NodeDB.t == 0).update(
            {"node_annot": VarAnnotation.FAKE}
        )
        session.commit()

    solve(config, overwrite=True, use_annotations=True)
    tracks_df_annot, _ = to_tracks_layer(config)

    assert len(tracks_df) > len(tracks_df_annot)
