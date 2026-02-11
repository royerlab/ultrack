from typing import Tuple

import numpy as np
import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from ultrack import MainConfig, add_new_node
from ultrack.core.database import LinkDB, NodeDB, OverlapDB
from ultrack.utils.constants import NO_PARENT


def _get_table_sizes(session: Session) -> Tuple[int, int, int]:
    return (
        session.query(NodeDB).count(),
        session.query(LinkDB).count(),
        session.query(OverlapDB).count(),
    )


class MinimalNode:
    """Minimal node-like object with only mask and bbox (simulating old pickle format)."""

    def __init__(self, mask, bbox):
        self.mask = mask
        self.bbox = bbox
        self.area = int(mask.sum())
        self.centroid = np.array(
            [mask.shape[0] // 2, mask.shape[1] // 2, mask.shape[2] // 2]
        )

    def IoU(self, other):
        """Simple IoU calculation for testing"""
        # Check if bboxes overlap
        ndim = len(self.bbox) // 2
        for i in range(ndim):
            if (
                self.bbox[i] >= other.bbox[i + ndim]
                or other.bbox[i] >= self.bbox[i + ndim]
            ):
                return 0.0
        return 0.1  # Return small overlap for testing


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


@pytest.mark.parametrize(
    "config_content",
    [
        {
            "data.database": "sqlite",
            "segmentation.n_workers": 4,
            "linking.n_workers": 4,
            "linking.max_distance": 500,
        },
    ],
    indirect=True,
)
def test_add_node_with_minimal_pickle(
    linked_database_mock_data: MainConfig,
) -> None:
    """Test that add_new_node works when existing nodes have pickle fields with only mask/bbox (no id)."""

    engine = create_engine(linked_database_mock_data.data_config.database_path)

    # First, manually add a node with minimal pickle (only mask/bbox, no id)
    minimal_mask = np.ones((5, 10, 10), dtype=bool)
    minimal_bbox = np.array([20, 0, 0, 25, 10, 10], dtype=int)
    minimal_node = MinimalNode(minimal_mask, minimal_bbox)

    with Session(engine) as session:
        # Manually insert a node with minimal pickle
        test_node = NodeDB(
            id=9999999,  # Use a unique ID
            t=0,
            t_node_id=999,
            t_hier_id=1,
            z=22,
            y=5,
            x=5,
            area=int(minimal_mask.sum()),
            pickle=minimal_node,  # Pickle without id attribute
        )
        session.add(test_node)
        session.commit()

    # Now add a new node that might overlap with the minimal node
    new_mask = np.ones((7, 12, 12), dtype=bool)
    new_bbox = np.array([22, 3, 3, 29, 15, 15], dtype=int)  # Overlaps with minimal_bbox

    with Session(engine) as session:
        n_nodes, n_links, n_overlaps = _get_table_sizes(session)

        # This should work even though minimal_node.pickle doesn't have an id
        add_new_node(
            linked_database_mock_data,
            0,
            new_mask,
            new_bbox,
        )

        new_n_nodes, new_n_links, new_n_overlaps = _get_table_sizes(session)

        # Query the new overlaps to verify ancestor_ids are valid (not -1)
        new_overlaps = session.query(OverlapDB).all()

    # Verify that nodes were added successfully
    assert new_n_nodes == n_nodes + 1
    # Overlaps may have been added if IoU > 0
    assert new_n_overlaps >= n_overlaps

    # Verify that all ancestor_ids in overlaps are valid (not NO_PARENT/-1)
    for overlap in new_overlaps:
        assert (
            overlap.ancestor_id != NO_PARENT
        ), f"Found overlap with ancestor_id={overlap.ancestor_id} (should not be NO_PARENT)"
        assert (
            overlap.ancestor_id > 0
        ), f"Found overlap with invalid ancestor_id={overlap.ancestor_id}"
