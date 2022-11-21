import pytest

from ultrack.config.config import MainConfig
from ultrack.core.database import LinkDB, NodeDB, OverlapDB, is_table_empty
from ultrack.core.segmentation.utils import clear_all_data


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
