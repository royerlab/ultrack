from pathlib import Path

import pytest

from ultrack import MainConfig, export_tracks_by_extension


def test_exporter(tracked_database_mock_data: MainConfig, tmp_path: Path) -> None:
    file_ext_list = [".xml", ".csv", ".zarr", ".dot", ".json", ".geff"]
    last_modified_time = {}
    for file_ext in file_ext_list:
        tmp_file = tmp_path / f"tracks{file_ext}"
        export_tracks_by_extension(tracked_database_mock_data, tmp_file)

        # assert file exists
        assert (tmp_path / f"tracks{file_ext}").exists()
        # assert file size is not zero
        assert (tmp_path / f"tracks{file_ext}").stat().st_size > 0

        # store last modified time
        last_modified_time[str(tmp_file)] = tmp_file.stat().st_mtime

    # loop again testing overwrite=False
    for file_ext in file_ext_list:
        tmp_file = tmp_path / f"tracks{file_ext}"
        try:
            export_tracks_by_extension(
                tracked_database_mock_data, tmp_file, overwrite=False
            )
            assert False, "FileExistsError should be raised"
        except FileExistsError:
            pass

    # loop again testing overwrite=True
    for file_ext in file_ext_list:
        tmp_file = tmp_path / f"tracks{file_ext}"
        export_tracks_by_extension(tracked_database_mock_data, tmp_file, overwrite=True)

        # assert file exists
        assert (tmp_path / f"tracks{file_ext}").exists()
        # assert file size is not zero
        assert (tmp_path / f"tracks{file_ext}").stat().st_size > 0

        assert last_modified_time[str(tmp_file)] != tmp_file.stat().st_mtime


def test_geff_zarr_extension_specific(
    tracked_database_mock_data: MainConfig, tmp_path: Path
) -> None:
    """Test specific functionality of .geff.zarr extension in exporter."""
    geff_file = tmp_path / "tracks.geff"

    # Test that .geff.zarr extension calls the to_geff function
    export_tracks_by_extension(tracked_database_mock_data, geff_file)

    # Check that file exists and has content
    assert geff_file.exists()
    assert geff_file.stat().st_size > 0

    # Test overwrite behavior
    with pytest.raises(FileExistsError, match="already exists"):
        export_tracks_by_extension(
            tracked_database_mock_data, geff_file, overwrite=False
        )

    # Test that overwrite=True works
    export_tracks_by_extension(tracked_database_mock_data, geff_file, overwrite=True)
    assert geff_file.exists()
    assert geff_file.stat().st_size > 0
