from pathlib import Path

from ultrack import MainConfig, export_tracks_by_extension


def test_exporter(tracked_database_mock_data: MainConfig, tmp_path: Path) -> None:
    file_ext_list = [".xml", ".csv", ".zarr", ".dot", ".json"]
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
