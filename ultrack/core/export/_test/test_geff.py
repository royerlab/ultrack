from pathlib import Path
from unittest.mock import patch

import pytest

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
    assert (output_file / "overlaps/ids").exists()
    assert (output_file / "overlaps/props").exists()
