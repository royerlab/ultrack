from typing import Dict, List, Tuple

import pandas as pd

from ultrack.config.dataconfig import DataConfig
from ultrack.core.export.utils import (
    add_track_ids_to_forest,
    inv_tracks_forest,
    solution_dataframe_from_sql,
)


def to_tracks_layer(
    data_config: DataConfig,
) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
    """Exports solution from database to napari tracks layer format.

    Parameters
    ----------
    data_config : DataConfig
        Data configuration parameters.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[int, List[int]]]
        Tracks dataframe and an lineage graph, mapping node_id -> parent_id.
    """
    df = solution_dataframe_from_sql(data_config.database_path)
    df = add_track_ids_to_forest(df)
    graph = inv_tracks_forest(df)

    data_dim = len(data_config.metadata["shape"])
    if data_dim == 4:
        columns = ["track_id", "t", "z", "y", "x"]
    elif data_dim == 3:
        columns["track_id", "t", "y", "x"]
    else:
        raise ValueError(
            f"Expected dataset with 3 or 4 dimensions, T(Z)YX. Found {data_dim}."
        )

    return df[columns], graph
