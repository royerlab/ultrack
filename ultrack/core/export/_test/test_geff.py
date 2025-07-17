import pytest
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import networkx as nx

from ultrack import MainConfig
from ultrack.core.export.geff import to_geff


def test_to_geff_basic_functionality(tracked_database_mock_data: MainConfig, tmp_path: Path):
    """Test basic functionality of to_geff function."""
    output_file = tmp_path / "test_tracks.geff.zarr"
    
    # Mock the geff.write_nx function to avoid actual file operations
    with patch('ultrack.core.export.geff.geff.write_nx') as mock_write_nx:
        to_geff(tracked_database_mock_data, output_file)
        
        # Verify that geff.write_nx was called with the correct arguments
        mock_write_nx.assert_called_once()
        args, kwargs = mock_write_nx.call_args
        
        # Check that the first argument is a NetworkX graph
        assert isinstance(args[0], nx.Graph)
        # Check that the second argument is the filename
        assert args[1] == output_file


def test_to_geff_file_overwrite_false(tracked_database_mock_data: MainConfig, tmp_path: Path):
    """Test that FileExistsError is raised when file exists and overwrite=False."""
    output_file = tmp_path / "test_tracks.geff.zarr"
    
    # Create a file that already exists
    output_file.touch()
    
    # Test that FileExistsError is raised when overwrite=False
    with pytest.raises(FileExistsError, match="already exists"):
        to_geff(tracked_database_mock_data, output_file, overwrite=False)


def test_to_geff_file_overwrite_true(tracked_database_mock_data: MainConfig, tmp_path: Path):
    """Test that function works when file exists and overwrite=True."""
    output_file = tmp_path / "test_tracks.geff.zarr"
    
    # Create a file that already exists
    output_file.touch()
    
    # Mock the geff.write_nx function
    with patch('ultrack.core.export.geff.geff.write_nx') as mock_write_nx:
        # This should not raise an error
        to_geff(tracked_database_mock_data, output_file, overwrite=True)
        
        # Verify that geff.write_nx was called
        mock_write_nx.assert_called_once()


def test_to_geff_with_string_filename(tracked_database_mock_data: MainConfig, tmp_path: Path):
    """Test that to_geff works with string filename."""
    output_file = str(tmp_path / "test_tracks.geff.zarr")
    
    # Mock the geff.write_nx function
    with patch('ultrack.core.export.geff.geff.write_nx') as mock_write_nx:
        to_geff(tracked_database_mock_data, output_file)
        
        # Verify that geff.write_nx was called with the correct arguments
        mock_write_nx.assert_called_once()
        args, kwargs = mock_write_nx.call_args
        
        # Check that the first argument is a NetworkX graph
        assert isinstance(args[0], nx.Graph)
        # Check that the second argument is the filename
        assert args[1] == output_file


def test_to_geff_calls_to_networkx(tracked_database_mock_data: MainConfig, tmp_path: Path):
    """Test that to_geff properly calls to_networkx to get the graph."""
    output_file = tmp_path / "test_tracks.geff.zarr"
    
    # Mock both functions to verify the call chain
    with patch('ultrack.core.export.geff.to_networkx') as mock_to_networkx, \
         patch('ultrack.core.export.geff.geff.write_nx') as mock_write_nx:
        
        # Set up the mock to return a simple graph
        mock_graph = nx.Graph()
        mock_to_networkx.return_value = mock_graph
        
        to_geff(tracked_database_mock_data, output_file)
        
        # Verify that to_networkx was called with the config
        mock_to_networkx.assert_called_once_with(tracked_database_mock_data)
        
        # Verify that geff.write_nx was called with the graph and filename
        mock_write_nx.assert_called_once_with(mock_graph, output_file)


def test_to_geff_default_overwrite_false(tracked_database_mock_data: MainConfig, tmp_path: Path):
    """Test that default overwrite parameter is False."""
    output_file = tmp_path / "test_tracks.geff.zarr"
    
    # Create a file that already exists
    output_file.touch()
    
    # Test that FileExistsError is raised when overwrite is not specified (default False)
    with pytest.raises(FileExistsError, match="already exists"):
        to_geff(tracked_database_mock_data, output_file)

def test_geff_correctness(tracked_database_mock_data: MainConfig, tmp_path: Path):
    """Test that the geff file is correct."""
    output_file = tmp_path / "test_tracks.geff.zarr"
    to_geff(tracked_database_mock_data, output_file)

    # Read the geff file
    import geff
    geff_nx = geff.read_nx(output_file)

    from ultrack.core.export import to_networkx
    ultrack_nx = to_networkx(tracked_database_mock_data)

    assert nx.is_isomorphic(geff_nx, ultrack_nx)