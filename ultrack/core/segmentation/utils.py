import logging
import warnings

import sqlalchemy as sqla
from numpy.typing import ArrayLike
from sqlalchemy.orm import Session

from ultrack.core.database import NodeDB, OverlapDB
from ultrack.core.linking.utils import clear_linking_data

LOG = logging.getLogger(__name__)


def check_array_chunk(array: ArrayLike) -> None:
    """Checks if chunked array has chunk size of 1 on time dimension."""
    if hasattr(array, "chunks"):
        chunk_shape = array.chunks
        if isinstance(chunk_shape[0], tuple):
            # sometimes the chunk shapes items are tuple, I don't know why
            chunk_shape = chunk_shape[0]
        if chunk_shape[0] != 1:
            warnings.warn(
                f"Array not chunked over time dimension. Found chunk of shape {array.chunks}."
                "Performance will be slower."
            )


def clear_segmentation_data(database_path: str) -> None:
    """Clears segmentation data its dependents (overlaps and links)."""
    clear_linking_data(database_path)

    LOG.info("Clearing segmentation database.")
    engine = sqla.create_engine(database_path)
    with Session(engine) as session:
        session.query(OverlapDB).delete()
        session.query(NodeDB).delete()
        session.commit()
