from typing import Tuple

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ultrack import MainConfig, add_new_node
from ultrack.core.database import LinkDB, NodeDB, OverlapDB


def _get_table_sizes(session: Session) -> Tuple[int, int, int]:
    return (
        session.query(NodeDB).count(),
        session.query(LinkDB).count(),
        session.query(OverlapDB).count(),
    )


@pytest.mark.parametrize(
    "config_content",
    [
        {
            "data.database": "sqlite",
            "segmentation.n_workers": 4,
            "linking.n_workers": 4,
            "linking.max_distance": 500,  # too big and ignored
        },
    ],
    indirect=True,
)
def test_clear_solution(
    linked_database_mock_data: MainConfig,
) -> None:

    mask = np.ones((7, 12, 12), dtype=bool)
    bbox = np.array([25, 4, 42, 32, 16, 54], dtype=int)

    engine = create_engine(linked_database_mock_data.data_config.database_path)
    with Session(engine) as session:
        n_nodes, n_links, n_overlaps = _get_table_sizes(session)

        add_new_node(
            linked_database_mock_data,
            0,
            mask,
            bbox,
        )

        new_n_nodes, new_n_links, new_n_overlaps = _get_table_sizes(session)

    assert new_n_nodes == n_nodes + 1
    assert new_n_overlaps > n_overlaps
    # could smaller than max neighbors because of radius
    assert (
        new_n_links == n_links + linked_database_mock_data.linking_config.max_neighbors
    )
