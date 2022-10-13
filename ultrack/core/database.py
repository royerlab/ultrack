import logging
from pathlib import Path

import sqlalchemy as sqla
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    PickleType,
    func,
)
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, declarative_base

from ultrack.config.dataconfig import DataConfig

# constant value to indicate it has no parent
NO_PARENT = -1

Base = declarative_base()

LOG = logging.getLogger(__name__)


class NodeDB(Base):
    __tablename__ = "nodes"
    t = Column(Integer, primary_key=True)
    id = Column(BigInteger, primary_key=True, unique=True)
    parent_id = Column(BigInteger, default=NO_PARENT)
    t_node_id = Column(Integer)
    t_hier_id = Column(Integer)
    z = Column(Float)
    y = Column(Float)
    x = Column(Float)
    area = Column(Integer)
    selected = Column(Boolean)
    pickle = Column(PickleType)


class OverlapDB(Base):
    __tablename__ = "overlaps"
    id = Column(Integer, primary_key=True, autoincrement=True)
    node_id = Column(BigInteger, ForeignKey(f"{NodeDB.__tablename__}.id"))
    ancestor_id = Column(BigInteger, ForeignKey(f"{NodeDB.__tablename__}.id"))


class LinkDB(Base):
    __tablename__ = "links"
    id = Column(Integer, primary_key=True, autoincrement=True)
    source_id = Column(BigInteger, ForeignKey(f"{NodeDB.__tablename__}.id"))
    target_id = Column(BigInteger, ForeignKey(f"{NodeDB.__tablename__}.id"))
    iou = Column(Float)


def maximum_time(data_config: DataConfig) -> int:
    """Returns the maximum `t` found in the `NodesDB`."""
    engine = sqla.create_engine(data_config.database_path)
    with Session(engine) as session:
        max_t = session.query(func.max(NodeDB.t)).scalar()

    LOG.info(f"Found max time = {max_t}")
    if max_t is None:
        raise ValueError(f"Dataset at {data_config.database_path} is empty.")

    return max_t


def is_table_empty(data_config: DataConfig, table: Base) -> bool:
    """Checks if table is empty."""
    url = make_url(data_config.database_path)
    if data_config.database == "sqlite" and not Path(url.database).exists():
        # avoids creating a database with create_engine call
        return True

    engine = sqla.create_engine(data_config.database_path)
    with Session(engine) as session:
        is_empty = (
            sqla.inspect(engine).has_table(table.__tablename__)
            and session.query(table).first() is None
        )
    return is_empty
