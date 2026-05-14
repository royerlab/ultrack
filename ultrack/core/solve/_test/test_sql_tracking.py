from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import pytest
import sqlalchemy as sqla
import toml
from sqlalchemy.orm import Session

from ultrack import solve, to_tracks_layer
from ultrack.config.config import MainConfig, load_config
from ultrack.core.database import Base, LinkDB, NodeDB, VarAnnotation
from ultrack.core.solve.sqltracking import SQLTracking
from ultrack.utils.constants import NO_PARENT
from ultrack.utils.data import make_config_content

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

    # every selected node's parent_id must reference another selected node;
    # regression guard for window-boundary phantoms in interleaved solving.
    selected = nodes[nodes["selected"]]
    parented = selected[selected["parent_id"] != NO_PARENT]
    parents_selected = nodes.loc[parented["parent_id"].values, "selected"].values
    assert np.all(parents_selected), (
        f"{(~parents_selected).sum()} selected nodes have parent_id pointing "
        "to a non-selected node (dangling parent)"
    )


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


def _make_minimal_tracking_config(tmp_path: Path, window_size: int, overlap_size: int):
    cfg = make_config_content(
        {
            "data.working_dir": str(tmp_path),
            "data.database": "sqlite",
            "tracking.window_size": window_size,
            "tracking.overlap_size": overlap_size,
        }
    )
    path = tmp_path / "config.toml"
    with open(path, mode="w") as f:
        toml.dump(cfg, f)
    return load_config(path)


def _seed_nodes(config: MainConfig, n_t: int) -> None:
    """Create empty NodeDB rows at every t in [0, n_t)."""
    engine = sqla.create_engine(config.data_config.database_path)
    Base.metadata.create_all(engine)
    rows = [
        dict(
            t=t,
            id=t,
            parent_id=NO_PARENT,
            hier_parent_id=NO_PARENT,
            t_node_id=0,
            t_hier_id=0,
            z=0.0,
            y=0.0,
            x=0.0,
            area=1,
            selected=False,
            pickle=None,
            features=None,
            node_prob=0.5,
        )
        for t in range(n_t)
    ]
    with Session(engine) as session:
        session.execute(sqla.insert(NodeDB), rows)
        session.commit()
    config.data_config.metadata_add({"shape": [n_t, 1, 1, 1]})


def _mark_selected(config: MainConfig, time_slices: Tuple[int, ...]) -> None:
    engine = sqla.create_engine(config.data_config.database_path)
    with Session(engine) as session:
        session.execute(
            sqla.update(NodeDB).where(NodeDB.t.in_(time_slices)).values(selected=True)
        )
        session.commit()


def test_compute_layout_first_pass(tmp_path: Path) -> None:
    """With no committed neighbours every side gets the full overlap."""
    config = _make_minimal_tracking_config(tmp_path, window_size=3, overlap_size=2)
    _seed_nodes(config, n_t=15)

    tracker = SQLTracking(config)

    layout = tracker._compute_layout(index=2)  # middle batch, inner [6, 8]
    assert layout.left_anchored is False
    assert layout.right_anchored is False
    assert (layout.solver_start, layout.solver_end) == (4, 10)
    assert (layout.commit_start, layout.commit_end) == (5, 9)

    first = tracker._compute_layout(index=0)  # leftmost batch, inner [0, 2]
    assert first.left_anchored is False
    assert first.right_anchored is False
    assert (first.solver_start, first.solver_end) == (0, 4)
    assert (first.commit_start, first.commit_end) == (0, 3)

    last_index = tracker.num_batches - 1
    last = tracker._compute_layout(index=last_index)
    assert last.right_anchored is False
    assert last.solver_end == tracker._max_t
    assert last.commit_end == tracker._max_t


def test_compute_layout_anchored(tmp_path: Path) -> None:
    """Committed neighbour slices shrink the layout to the inner range."""
    config = _make_minimal_tracking_config(tmp_path, window_size=3, overlap_size=2)
    _seed_nodes(config, n_t=15)
    # batch index 2 has inner [6, 8]; mark its boundaries to simulate that
    # both neighbouring batches have already committed.
    _mark_selected(config, time_slices=(5, 9))

    tracker = SQLTracking(config)
    layout = tracker._compute_layout(index=2)
    assert layout.left_anchored is True
    assert layout.right_anchored is True
    assert (layout.solver_start, layout.solver_end) == (6, 8)
    assert (layout.commit_start, layout.commit_end) == (6, 8)


def test_compute_layout_mixed_anchoring(tmp_path: Path) -> None:
    """Only one side anchored: that side shrinks, the other keeps the overlap."""
    config = _make_minimal_tracking_config(tmp_path, window_size=3, overlap_size=2)
    _seed_nodes(config, n_t=15)
    _mark_selected(config, time_slices=(5,))  # only left neighbour committed

    tracker = SQLTracking(config)
    layout = tracker._compute_layout(index=2)
    assert layout.left_anchored is True
    assert layout.right_anchored is False
    assert layout.solver_start == 6
    assert layout.commit_start == 6
    assert layout.solver_end == 10
    assert layout.commit_end == 9


def test_is_committed_at_out_of_range(tmp_path: Path) -> None:
    """Out-of-range times return False without touching the DB."""
    config = _make_minimal_tracking_config(tmp_path, window_size=3, overlap_size=2)
    _seed_nodes(config, n_t=5)

    tracker = SQLTracking(config)
    assert tracker._is_committed_at(-1) is False
    assert tracker._is_committed_at(tracker._max_t + 1) is False
