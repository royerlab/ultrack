import logging

import sqlalchemy as sqla
from sqlalchemy.orm import Session

from ultrack.core.database import LinkDB
from ultrack.core.solve.sqltracking import SQLTracking

LOG = logging.getLogger(__name__)


def clear_linking_data(database_path: str) -> None:
    """Clears linking data."""
    SQLTracking.clear_solution_from_database(database_path)

    LOG.info("Clearing links database.")
    engine = sqla.create_engine(database_path)
    with Session(engine) as session:
        session.query(LinkDB).delete()
        session.commit()
