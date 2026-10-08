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


def _read_validated(output_file: Path):
    import geff
    from geff.validate.data import ValidationConfig

    geff.validate_structure(output_file)
    return geff.read(
        output_file,
        data_validation=ValidationConfig(graph=True, tracklet=True, lineage=True),
    )


@pytest.mark.parametrize("zarr_format", [2, 3])
@pytest.mark.parametrize(
    "timelapse_mock_data",
    [{"n_dim": 2, "size": 128}, {"n_dim": 3, "size": 64}],
    indirect=True,
)
def test_geff_tracks_and_axes(
    tracked_database_mock_data: MainConfig,
    tmp_path: Path,
    zarr_format: int,
) -> None:
    """Test track ids, axes and spec validity for 2D/3D data and zarr v2/v3."""
    from ultrack.core.export import to_tracks_layer

    config = tracked_database_mock_data
    n_spatial = len(config.data_config.metadata["shape"]) - 1
    spatial_axes = ["z", "y", "x"][-n_spatial:]
    scale = [0.5, 0.25, 0.125][-n_spatial:]

    output_file = tmp_path / "tracks.geff.zarr"
    to_geff(
        config,
        output_file,
        zarr_format=zarr_format,
        segmentation_path="../segments.zarr",
        scale=scale,
        spatial_unit="micrometer",
        time_scale=5.0,
        time_unit="minute",
    )

    assert zarr.open_group(output_file, mode="r").metadata.zarr_format == zarr_format

    graph, metadata = _read_validated(output_file)

    # axes
    assert [ax.name for ax in metadata.axes] == ["t", *spatial_axes]
    t_axis = metadata.axes[0]
    assert (t_axis.type, t_axis.unit, t_axis.scale, t_axis.scaled_unit) == (
        "time",
        "frame",
        5.0,
        "minute",
    )
    for ax, s in zip(metadata.axes[1:], scale):
        assert (ax.type, ax.unit, ax.scale, ax.scaled_unit) == (
            "space",
            "pixel",
            s,
            "micrometer",
        )

    # tracks metadata
    assert metadata.track_node_props == {
        "tracklet": "track_id",
        "lineage": "lineage_id",
    }
    assert len(metadata.related_objects) == 1
    related = metadata.related_objects[0]
    assert (related.type, related.path, related.node_prop) == (
        "labels",
        "../segments.zarr",
        "seg_id",
    )
    assert metadata.extra["ultrack"]["overlaps"]["path"] == "overlaps/ids"

    # node properties
    node_attrs = graph.nodes[next(iter(graph.nodes))]
    for prop in ("t", *spatial_axes, "track_id", "parent_track_id", "lineage_id"):
        assert prop in node_attrs, f"Missing node property: {prop}"
    if n_spatial == 2:
        assert "z" not in node_attrs
        assert "z_shift" not in node_attrs
    assert node_attrs["mask"].dtype == bool
    assert node_attrs["mask"].ndim == n_spatial
    assert len(node_attrs["bbox"]) == 2 * n_spatial

    # track ids are identical to `to_tracks_layer`
    tracks_df, _ = to_tracks_layer(config)
    tracks_df = tracks_df.set_index("id")
    assert len(tracks_df) == graph.number_of_nodes()
    for node_id, attrs in graph.nodes(data=True):
        row = tracks_df.loc[node_id]
        assert attrs["track_id"] == row["track_id"]
        assert attrs["seg_id"] == row["track_id"]
        assert attrs["parent_track_id"] == row["parent_track_id"]


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_geff_lineages_with_divisions(
    tracked_cell_division_mock_data: MainConfig,
    tmp_path: Path,
    zarr_format: int,
) -> None:
    """Test tracklets and lineages of a dividing cell."""
    import networkx as nx

    output_file = tmp_path / "tracks.geff.zarr"
    to_geff(tracked_cell_division_mock_data, output_file, zarr_format=zarr_format)

    graph, _ = _read_validated(output_file)

    # the mock data has divisions, otherwise the test is vacuous
    assert any(graph.out_degree(n) > 1 for n in graph.nodes)

    for component in nx.weakly_connected_components(graph):
        root_track_ids = {
            graph.nodes[n]["track_id"] for n in component if graph.in_degree(n) == 0
        }
        assert len(root_track_ids) == 1
        (root_track_id,) = root_track_ids
        for n in component:
            assert graph.nodes[n]["lineage_id"] == root_track_id
            if graph.nodes[n]["track_id"] == root_track_id:
                assert graph.nodes[n]["parent_track_id"] == -1

    for source, target in graph.edges:
        if graph.out_degree(source) > 1:
            # new tracklet after division
            assert (
                graph.nodes[target]["parent_track_id"]
                == graph.nodes[source]["track_id"]
            )
        else:
            assert graph.nodes[target]["track_id"] == graph.nodes[source]["track_id"]


@pytest.mark.parametrize("zarr_format", [2, 3])
def test_geff_without_masks_and_overlaps(
    tracked_database_mock_data: MainConfig,
    tmp_path: Path,
    zarr_format: int,
) -> None:
    """Test that masks and overlaps can be omitted."""
    output_file = tmp_path / "tracks.geff.zarr"
    to_geff(
        tracked_database_mock_data,
        output_file,
        zarr_format=zarr_format,
        include_masks=False,
        include_overlaps=False,
    )

    graph, metadata = _read_validated(output_file)

    assert "mask" not in metadata.node_props_metadata
    assert "mask" not in graph.nodes[next(iter(graph.nodes))]
    assert metadata.related_objects is None
    assert "ultrack" not in metadata.extra

    store = zarr.open_group(output_file, mode="r")
    assert "nodes/props/mask" not in store
    assert "overlaps" not in store


@pytest.mark.parametrize(
    "timelapse_mock_data",
    [{"n_dim": 2, "size": 128}, {"n_dim": 3, "size": 64}],
    indirect=True,
)
def test_geff_seg_id_matches_painted_labels(
    tracked_database_mock_data: MainConfig,
    tmp_path: Path,
) -> None:
    """Test that `seg_id` is the label painted by `Tracker.to_zarr` at each node."""
    import numpy as np
    from geff.validate.segmentation import has_seg_ids_at_coords

    from ultrack import Tracker
    from ultrack.core.tracker import TrackerStatus

    tracker = Tracker(tracked_database_mock_data)
    tracker.status |= TrackerStatus.SOLVED

    segments_path = tmp_path / "segments.zarr"
    segments = tracker.to_zarr(store_or_path=segments_path)[...]

    output_file = tmp_path / "tracks.geff.zarr"
    tracker.to_geff(output_file, segmentation_path="../segments.zarr", zarr_format=3)

    graph, metadata = _read_validated(output_file)
    spatial_axes = [ax.name for ax in metadata.axes[1:]]

    coords, seg_ids = [], []
    for _, attrs in graph.nodes(data=True):
        t = attrs["t"]
        seg_id = attrs["seg_id"]
        n_spatial = len(spatial_axes)
        start = attrs["bbox"][:n_spatial]
        end = attrs["bbox"][n_spatial:]
        crop = segments[(t,) + tuple(slice(s, e) for s, e in zip(start, end))]
        # every pixel of the node mask is painted with its seg_id
        assert np.all(crop[attrs["mask"]] == seg_id)

        centroid = tuple(int(round(attrs[ax])) for ax in spatial_axes)
        if attrs["mask"][tuple(c - s for c, s in zip(centroid, start))]:
            coords.append((t, *centroid))
            seg_ids.append(seg_id)

    assert len(seg_ids) > 0
    valid, errors = has_seg_ids_at_coords(segments, coords, seg_ids)
    assert valid, errors

    # labels and seg_ids are the same set at every time point
    for t in range(segments.shape[0]):
        painted = set(np.unique(segments[t])) - {0}
        exported = {a["seg_id"] for _, a in graph.nodes(data=True) if a["t"] == t}
        assert painted == exported


def test_geff_scale_validation(
    tracked_database_mock_data: MainConfig, tmp_path: Path
) -> None:
    """Test inconsistent scale / unit parameters."""
    output_file = tmp_path / "tracks.geff.zarr"
    with pytest.raises(ValueError, match="one value per spatial axis"):
        to_geff(tracked_database_mock_data, output_file, scale=[1.0, 1.0])
    with pytest.raises(ValueError, match="`scale` must be provided"):
        to_geff(tracked_database_mock_data, output_file, spatial_unit="micrometer")
    with pytest.raises(ValueError, match="`time_scale` must be provided"):
        to_geff(tracked_database_mock_data, output_file, time_unit="minute")
