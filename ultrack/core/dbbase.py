from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Float,
    ForeignKey,
    Integer,
    PickleType,
)
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
