import logging

import networkx as nx
import pandas as pd

from ultrack.config.config import MainConfig
from ultrack.core.export.tracks_layer import to_tracks_layer
from ultrack.tracks.graph import _create_tracks_forest
from ultrack.utils.constants import NO_PARENT

LOG = logging.getLogger(__name__)


def tracks_layer_to_networkx(
    tracks_df: pd.DataFrame,
    children_to_parent: bool = False,
) -> nx.DiGraph:
    """
    Convert napari tracks layer tracks dataframe to networkx directed graph.
    By default, the edges are the parent to child relationships.

    Parameters
    ----------
    config : MainConfig
        Configuration parameters.
    children_to_parent : bool
        If set, edges encode child to parent relationships.

    Returns
    -------
    nx.DiGraph
        Networkx graph.
    """
    graph = nx.DiGraph()

    if "id" not in tracks_df.columns:
        LOG.warning(
            "`id` not found in tracks dataframe. Assuming `id` is dataframe index."
        )
    else:
        tracks_df = tracks_df.set_index("id")

    for index, row in tracks_df.to_dict(orient="index").items():
        graph.add_node(index, **row)

    if "parent_id" not in tracks_df.columns:
        LOG.warning(
            "Parent id not found in tracks dataframe. Divisions won't be exported."
        )
        for _, group in tracks_df.groupby("track_id"):
            for i in range(len(group) - 1):
                if children_to_parent:
                    graph.add_edge(group.index[i + 1], group.index[i])
                else:
                    graph.add_edge(group.index[i], group.index[i + 1])

    else:
        dict_graph = _create_tracks_forest(
            tracks_df.index.values,
            tracks_df["parent_id"].to_numpy(int),
        )
        dict_graph.pop(NO_PARENT)

        for parent, children in dict_graph.items():
            for child in children:
                if children_to_parent:
                    graph.add_edge(child, parent)
                else:
                    graph.add_edge(parent, child)

    return graph


def to_networkx(
    config: MainConfig,
    children_to_parent: bool = False,
) -> nx.DiGraph:
    """
    Convert solution from database to networkx directed graph.
    By default, the edges are the parent to child relationships.

    Parameters
    ----------
    config : MainConfig
        Configuration parameters.
    children_to_parent : bool
        If set, edges encode child to parent relationships.

    Returns
    -------
    nx.DiGraph
        Networkx graph.
    """
    df, _ = to_tracks_layer(config)
    return tracks_layer_to_networkx(df, children_to_parent)
