import logging
from typing import Sequence, Union

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


def add_flow(
    data_config: DataConfig,
    vector_field: Union[ArrayLike, Sequence[ArrayLike]],
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
        Vector field arrays. Each array per coordinate or a single (T, D, (Z), Y, X)-array.
    """
    LOG.info("Adding shift (vector field) to nodes")

    if isinstance(vector_field, Sequence):
        assert all(v.shape == vector_field[0].shape for v in vector_field)
        ndim = vector_field[0].ndim
        nvecs = len(vector_field)
        vec_shape = np.asarray(vector_field[0].shape)
        is_sequence = True
    else:
        ndim = vector_field.ndim - 1
        nvecs = vector_field.shape[1]
        vec_shape = np.asarray([vector_field.shape[0], *vector_field.shape[2:]])
        is_sequence = False

    shape = np.asarray(data_config.metadata["shape"])
    if len(shape) != ndim:
        raise ValueError(
            "Original data shape and vector field must have same number of dimensions (ignoring channels)."
            f" Found {len(shape)} and {ndim}."
        )

    columns = ["x_shift", "y_shift", "z_shift"]
    scaling = (vec_shape[1:] - 1) / (shape[1:] - 1)
    coordinate_columns = ["z", "y", "x"][-len(scaling) :]
    vec_index_iterator = reversed(range(nvecs))

    engine = sqla.create_engine(data_config.database_path)

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
        # reversed because z could be missing
        for v, colname in zip(vec_index_iterator, columns):
            if is_sequence:
                df[colname] = np.asarray(vector_field[v][t])[
                    coords
                ]  # asarray due lazy loading formats (e.g. dask)
            else:
                df[colname] = np.asarray(vector_field[t, v])[coords]

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
