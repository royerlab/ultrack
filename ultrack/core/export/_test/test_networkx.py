import pandas as pd
import pytest

from ultrack.core.export import tracks_layer_to_networkx
from ultrack.utils.constants import NO_PARENT


@pytest.mark.parametrize("children_to_parent", [True, False])
def test_to_networkx(
    children_to_parent: bool,
) -> None:
    tracks_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "parent_id": [NO_PARENT, 1, 2, 2],
            "track_id": [1, 1, 2, 3],
            "t": [0, 1, 2, 2],
            "z": [0, 0, 0, 0],
            "y": [10, 20, 30, 10],
            "x": [1, 2, 3, 1],
        }
    )
    tracks_df.set_index("id", drop=False, inplace=True)

    graph = tracks_layer_to_networkx(tracks_df, children_to_parent=children_to_parent)

    assert len(graph.nodes) == tracks_df.shape[0]
    assert len(graph.edges) == tracks_df.shape[0] - 1

    for node, data in graph.nodes(data=True):
        assert node in tracks_df.index
        for key, value in data.items():
            assert value == tracks_df.loc[node, key]

    if children_to_parent:
        for edge in [(2, 1), (3, 2), (4, 2)]:
            assert edge in graph.edges
    else:
        for edge in [(1, 2), (2, 3), (2, 4)]:
            assert edge in graph.edges


def test_to_networkx_without_parent_id() -> None:
    tracks_df = pd.DataFrame(
        {
            "id": [1, 2, 3, 4],
            "track_id": [1, 1, 2, 3],
            "t": [0, 1, 2, 2],
            "z": [0, 0, 0, 0],
            "y": [10, 20, 30, 10],
            "x": [1, 2, 3, 1],
        }
    )
    tracks_df.set_index("id", drop=False, inplace=True)

    graph = tracks_layer_to_networkx(tracks_df)

    assert len(graph.nodes) == tracks_df.shape[0]
    assert len(graph.edges) == 1

    for node in graph.nodes:
        assert node in tracks_df.index

    assert (1, 2) in graph.edges
