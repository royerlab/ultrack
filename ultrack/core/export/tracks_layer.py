from typing import Dict, List, Tuple

import pandas as pd

from ultrack.config.config import MainConfig
from ultrack.core.export.utils import solution_dataframe_from_sql
from ultrack.tracks.graph import add_track_ids_to_tracks_df, inv_tracks_df_forest


def to_tracks_layer(
    config: MainConfig,
    include_parents: bool = True,
    include_node_ids: bool = True,
) -> Tuple[pd.DataFrame, Dict[int, List[int]]]:
    """Exports solution from database to napari tracks layer format.

    Parameters
    ----------
    config : MainConfig
        Configuration parameters.
    include_parents : bool
        Flag to include parents track id for each track id.
    include_ids : bool
        Flag to include node ids for each unit.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[int, List[int]]]
        Tracks dataframe and an lineage graph, mapping node_id -> parent_id.
    """
    df = solution_dataframe_from_sql(config.data_config.database_path)
    df = add_track_ids_to_tracks_df(df)
    df.sort_values(by=["track_id", "t"], inplace=True)

    graph = inv_tracks_df_forest(df)

    data_dim = len(config.data_config.metadata["shape"])
    if data_dim == 4:
        columns = ["track_id", "t", "z", "y", "x"]
    elif data_dim == 3:
        columns = ["track_id", "t", "y", "x"]
    else:
        raise ValueError(
            f"Expected dataset with 3 or 4 dimensions, T(Z)YX. Found {data_dim}."
        )

    if include_node_ids:
        df["id"] = df.index
        columns.append("id")

    if include_parents:
        columns.append("parent_track_id")

        if include_node_ids:
            columns.append("parent_id")

    return df[columns], graph
