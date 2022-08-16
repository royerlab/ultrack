from pathlib import Path

from sqlalchemy import BigInteger, Boolean, Column, Float, Integer, PickleType
from sqlalchemy.orm import declarative_base

# constant value to indicate it has no parent
NO_PARENT = -1

Base = declarative_base()


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


def get_database_path(working_dir: Path, database: str) -> str:
    """Returns database path given working directory and database type."""
    if database.lower() == "sqlite":
        return f"sqlite:///{working_dir.absolute()}/data.db"
    else:
        raise NotImplementedError
