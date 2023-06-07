import logging
from typing import Sequence

import numpy as np
import pandas as pd
import sqlalchemy as sqla
from numpy.typing import ArrayLike
from sqlalchemy.orm import Session
from tqdm import tqdm

from ultrack.config import DataConfig
from ultrack.core.database import NodeDB

logging.basicConfig()
logging.getLogger("sqlachemy.engine").setLevel(logging.INFO)

LOG = logging.getLogger(__name__)


def add_shift(
    data_config: DataConfig,
    vector_field: Sequence[ArrayLike],
) -> None:
    """
    Adds vector field (coordinate shift) data into nodes.
    If there are fewer vector fields than dimensions, the last dimensions from (z,y,x) have priority.
    For example, if 2 vector fields are provided for a 3D data, only (y, x) are updated.
    Vector field shape, except `t`, can be different from the original image.
    When this happens, the indexing is done mapping the position and rounding.

    Parameters
    ----------
    data_config : DataConfig
        Data configuration parameters.
    vector_field : Sequence[ArrayLike]
        Vector field arrays. Each array per coordinate.
    """
    LOG.info("Adding shift (vector field) to nodes")

    assert all(v.shape == vector_field[0].shape for v in vector_field)

    shape = np.asarray(data_config.metadata["shape"])
    if len(shape) != vector_field[0].ndim:
        raise ValueError(
            "Original data shape and vector field must have same number of dimensions."
            f" Found {len(shape)} and {vector_field[0].ndim}."
        )

    engine = sqla.create_engine(data_config.database_path)

    vec_shape = np.asarray(vector_field[0].shape)
    scaling = (vec_shape[1:] - 1) / (shape[1:] - 1)

    columns = ["x_shift", "y_shift", "z_shift"]
    coordinate_columns = ["z", "y", "x"][-len(scaling) :]

    for t in tqdm(range(shape[0])):
        with Session(engine) as session:
            query = session.query(NodeDB.id, NodeDB.z, NodeDB.y, NodeDB.x).where(
                NodeDB.t == t
            )
            df = pd.read_sql_query(query.statement, session.bind, index_col="id")

        if len(df) == 0:
            LOG.warning(f"No node found at time point {t}.")
            continue

        coords = df[coordinate_columns].to_numpy()
        coords = np.round(coords * scaling).astype(int)
        coords = np.minimum(
            np.maximum(0, coords), vec_shape[1:] - 1
        )  # truncating boundary
        coords = tuple(coords.T)

        # default value
        df[columns] = 0.0
        for vec, colname in zip(reversed(vector_field), columns):
            df[colname] = np.asarray(vec[t])[
                coords
            ]  # asarray due lazy loading formats (e.g. dask)

        df["node_id"] = df.index
        with Session(engine) as session:
            statement = (
                sqla.update(NodeDB)
                .where(NodeDB.id == sqla.bindparam("node_id"))
                .values(
                    z_shift=sqla.bindparam("z_shift"),
                    y_shift=sqla.bindparam("y_shift"),
                    x_shift=sqla.bindparam("x_shift"),
                )
            )
            session.execute(
                statement,
                df[["node_id"] + columns].to_dict("records"),
                execution_options={"synchronize_session": False},
            )
            session.commit()
