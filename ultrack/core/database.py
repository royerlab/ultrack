import enum
import logging
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import pandas as pd
import sqlalchemy as sqla
from numpy.typing import ArrayLike
from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    Enum,
    Float,
    ForeignKey,
    Integer,
    PickleType,
    func,
)
from sqlalchemy.engine import make_url
from sqlalchemy.orm import Session, declarative_base

from ultrack.config.dataconfig import DatabaseChoices, DataConfig

# constant value to indicate it has no parent
NO_PARENT = -1

Base = declarative_base()

LOG = logging.getLogger(__name__)


class MaybePickleType(PickleType):
    """Only (un)pickle if value is not bytes."""

    cache_ok = True

    def bind_processor(self, dialect):
        processor = super().bind_processor(dialect)

        def _process(value):
            if isinstance(value, (bytes, memoryview)):
                return value
            return processor(value)

        return _process

    def result_processor(self, dialect, coltype):
        processor = super().result_processor(dialect, coltype)

        def _process(value):
            if not isinstance(value, (bytes, memoryview)):
                return value
            return processor(value)

        return _process


class NodeSegmAnnotation(enum.IntEnum):
    UNKNOWN = 0
    CORRECT = 1
    UNDERSEGMENTED = 2
    OVERSEGMENTED = 3


class VarAnnotation(enum.IntEnum):
    UNKNOWN = 0
    REAL = 1
    FAKE = 2


class NodeDB(Base):
    __tablename__ = "nodes"
    t = Column(Integer, primary_key=True)
    id = Column(BigInteger, primary_key=True, unique=True)
    parent_id = Column(BigInteger, default=NO_PARENT)
    # hierarchy parent id matches to `id` column.
    hier_parent_id = Column(BigInteger, default=NO_PARENT)
    t_node_id = Column(Integer)
    t_hier_id = Column(Integer)
    z = Column(Float)
    y = Column(Float)
    x = Column(Float)
    z_shift = Column(Float, default=0.0)
    y_shift = Column(Float, default=0.0)
    x_shift = Column(Float, default=0.0)
    area = Column(Integer)
    frontier = Column(Float, default=-1.0)
    selected = Column(Boolean, default=False)
    pickle = Column(MaybePickleType)
    features = Column(MaybePickleType, default=None)
    segm_annot = Column(Enum(NodeSegmAnnotation), default=NodeSegmAnnotation.UNKNOWN)
    node_annot = Column(Enum(VarAnnotation), default=VarAnnotation.UNKNOWN)
    appear_annot = Column(Enum(VarAnnotation), default=VarAnnotation.UNKNOWN)
    disappear_annot = Column(Enum(VarAnnotation), default=VarAnnotation.UNKNOWN)
    division_annot = Column(Enum(VarAnnotation), default=VarAnnotation.UNKNOWN)


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
    weight = Column(Float)
    annotation = Column(Enum(VarAnnotation), default=VarAnnotation.UNKNOWN)


class GroundTruthDB(Base):
    __tablename__ = "ground_truth"
    t = Column(Integer, primary_key=True)
    id = Column(Integer, primary_key=True, autoincrement=True)
    node_id = Column(BigInteger, ForeignKey(f"{NodeDB.__tablename__}.id"))
    weight = Column(Float)
    pickle = Column(MaybePickleType)
    label = Column(Integer)


def maximum_time_from_database(data_config: DataConfig) -> int:
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
    if (
        data_config.database == DatabaseChoices.sqlite.value
        and not Path(url.database).exists()
    ):
        # avoids creating a database with create_engine call
        return True

    engine = sqla.create_engine(data_config.database_path)
    with Session(engine) as session:
        is_empty = (
            not sqla.inspect(engine).has_table(table.__tablename__)
            or session.query(table).first() is None
        )
    return is_empty


def set_node_values(
    data_config: DataConfig,
    node_id: int,
    **kwargs,
) -> None:
    """Set arbitrary values to a node in the database given its `node_id`.

    Parameters
    ----------
    data_config : DataConfig
        Data configuration parameters.
    node_id : int
        Node database index.
    annot : NodeAnnotation
        Node annotation.
    """
    if hasattr(node_id, "item"):
        node_id = node_id.item()

    engine = sqla.create_engine(data_config.database_path)
    with Session(engine) as session:
        stmt = sqla.update(NodeDB).where(NodeDB.id == node_id).values(**kwargs)
        session.execute(stmt)
        session.commit()


def get_node_values(
    data_config: DataConfig,
    indices: Optional[Union[int, ArrayLike]],
    values: Union[Column, List[Column]],
) -> List[Any]:
    """Get the annotation of `node_id`.

    Parameters
    ----------
    data_config : DataConfig
        Data configuration parameters.
    indices : int
        Node database indices.
    values : List[Column]
        List of columns to be queried.
    """
    if not isinstance(values, List):
        values = [values]

    values.insert(0, NodeDB.id)

    is_scalar = False
    if isinstance(indices, int):
        indices = [indices]
        is_scalar = True

    elif isinstance(indices, np.ndarray):
        indices = indices.astype(int).tolist()

    engine = sqla.create_engine(data_config.database_path)
    with Session(engine) as session:
        query = session.query(*values)

        if indices is not None:
            query = query.where(NodeDB.id.in_(indices))

        df = pd.read_sql_query(query.statement, session.bind, index_col="id")

    if indices is not None and len(df) != len(indices):
        raise ValueError(
            f"Query returned {len(df)} rows, expected {len(indices)}."
            "\nCheck if node_id exists in database."
        )

    df = df.squeeze()
    if is_scalar:
        try:
            df = df.item()
        except ValueError:
            pass

    return df


def clear_all_data(database_path: str) -> None:
    """Clears all data from database"""
    LOG.info("Clearing all databases.")
    engine = sqla.create_engine(database_path)
    Base.metadata.drop_all(engine)
