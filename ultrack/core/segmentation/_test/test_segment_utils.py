import pytest
import sqlalchemy as sqla
from sqlalchemy.orm import Session

from ultrack.config.config import MainConfig
from ultrack.core.database import LinkDB, NodeDB, OverlapDB
from ultrack.core.segmentation.utils import clear_segmentation_data


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
def test_clear_segmentation_data(
    tracked_database_mock_data: MainConfig,
) -> None:
    database_path = tracked_database_mock_data.data_config.database_path

    clear_segmentation_data(database_path)

    engine = sqla.create_engine(database_path)
    with Session(engine) as session:
        assert session.query(LinkDB).count() == 0
        assert session.query(OverlapDB).count() == 0
        assert session.query(NodeDB).count() == 0
