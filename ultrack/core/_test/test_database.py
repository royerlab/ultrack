import pytest

from ultrack.config.config import MainConfig
from ultrack.core.database import (
    LinkDB,
    NodeDB,
    OverlapDB,
    clear_all_data,
    get_node_values,
    is_table_empty,
    set_node_values,
)
from ultrack.core.segmentation.processing import _generate_id


@pytest.mark.parametrize(
    "config_content",
    [
        {
            "segmentation.n_workers": 4,
            "linking.n_workers": 4,
        }
    ],
    indirect=True,
)
def test_clear_all_data(
    tracked_database_mock_data: MainConfig,
) -> None:
    data_config = tracked_database_mock_data.data_config

    clear_all_data(data_config.database_path)

    assert is_table_empty(data_config, NodeDB)
    assert is_table_empty(data_config, OverlapDB)
    assert is_table_empty(data_config, LinkDB)


def test_set_get_node_values(
    segmentation_database_mock_data: MainConfig,
) -> None:

    index = _generate_id(1, 1, 1_000_000)

    set_node_values(
        segmentation_database_mock_data.data_config,
        index,
        area=0,
    )

    value = get_node_values(
        segmentation_database_mock_data.data_config,
        index,
        [NodeDB.area],
    )

    assert value == 0
