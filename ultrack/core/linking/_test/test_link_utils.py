import pytest
import sqlalchemy as sqla
from sqlalchemy.orm import Session

from ultrack.config.config import MainConfig
from ultrack.core.database import LinkDB, NodeDB
from ultrack.core.linking.utils import clear_linking_data
from ultrack.utils.constants import NO_PARENT


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
def test_clear_linking_data(
    tracked_database_mock_data: MainConfig,
) -> None:
    database_path = tracked_database_mock_data.data_config.database_path

    clear_linking_data(database_path)

    engine = sqla.create_engine(database_path)
    with Session(engine) as session:
        assert session.query(LinkDB).count() == 0
        assert session.query(NodeDB).where(NodeDB.selected).count() == 0
        assert session.query(NodeDB).where(NodeDB.parent_id != NO_PARENT).count() == 0
        # not checking overlap because our mock data doesn't have any overlap
        assert session.query(NodeDB).count() > 0
