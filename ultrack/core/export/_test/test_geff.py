from pathlib import Path
from unittest.mock import patch

import pytest
import zarr

from ultrack import MainConfig
from ultrack.core.export.geff import to_geff


def test_to_geff_file_overwrite_false(
    tracked_database_mock_data: MainConfig, tmp_path: Path
) -> None:
    """Test that FileExistsError is raised when file exists and overwrite=False."""
    output_file = tmp_path / "test_tracks.geff.zarr"

    # Create a file that already exists
    output_file.touch()

    # Test that FileExistsError is raised when overwrite=False
    with pytest.raises(FileExistsError, match="already exists"):
        to_geff(tracked_database_mock_data, output_file, overwrite=False)


def test_to_geff_file_overwrite_true(
    tracked_database_mock_data: MainConfig, tmp_path: Path
):
    """Test that function works when file exists and overwrite=True."""
    output_file = tmp_path / "test_tracks.geff.zarr"

    # Create a file that already exists
    output_file.mkdir()

    # Mock the geff.write_nx function
    with patch("ultrack.core.export.geff.write_arrays") as mock_write_nx:
        # This should not raise an error
        to_geff(tracked_database_mock_data, output_file, overwrite=True)

        # Verify that geff.write_nx was called
        mock_write_nx.assert_called_once()


def test_geff_correctness(
    tracked_database_mock_data: MainConfig, tmp_path: Path
) -> None:
    """Test that the geff file is correct."""
    output_file = tmp_path / "test_tracks.geff.zarr"
    to_geff(tracked_database_mock_data, output_file)

    import geff
    import networkx as nx

    from ultrack.core.export import to_networkx

    geff_nx, _ = geff.read(output_file, backend="networkx")

    solution_geff_nx = nx.subgraph_view(
        geff_nx,
        filter_node=lambda n: geff_nx.nodes[n]["solution"],
        filter_edge=lambda s, t: geff_nx.edges[s, t]["solution"],
    )

    ultrack_nx = to_networkx(tracked_database_mock_data)

    assert nx.is_isomorphic(solution_geff_nx, ultrack_nx)

    # Check node properties are exported
    sample_node = next(iter(geff_nx.nodes))
    node_attrs = geff_nx.nodes[sample_node]
    for prop in ("t", "z", "y", "x", "area", "solution"):
        assert prop in node_attrs, f"Missing node property: {prop}"

    # Check edge properties are exported
    sample_edge = next(iter(geff_nx.edges))
    edge_attrs = geff_nx.edges[sample_edge]
    assert "solution" in edge_attrs

    # Check mask/bbox and overlaps are stored in the zarr store
    store = zarr.open(output_file, mode="r")
    assert "nodes/props/mask" in store
    assert "nodes/props/bbox" in store
    assert "overlaps/ids" in store
    assert "overlaps/props" in store


def test_geff_only_exports_solution_nodes_and_edges(
    tracked_database_mock_data: MainConfig, tmp_path: Path
) -> None:
    """Test that only solution nodes and edges are exported to geff."""
    import sqlalchemy as sqla
    from sqlalchemy.orm import Session

    from ultrack.core.database import NodeDB

    config = tracked_database_mock_data
    output_file = tmp_path / "test_tracks.geff.zarr"
    to_geff(config, output_file)

    engine = sqla.create_engine(config.data_config.database_path)
    with Session(engine) as session:
        n_selected = session.query(NodeDB).filter(NodeDB.selected).count()
        n_total = session.query(NodeDB).count()

    # Ensure the fixture has non-selected nodes to make this test meaningful
    assert n_selected < n_total, "Mock data has no non-selected nodes; test is vacuous"

    store = zarr.open(output_file, mode="r")
    n_exported = store["nodes/ids"].shape[0]
    assert n_exported == n_selected
