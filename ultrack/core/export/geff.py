import shutil
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union
from warnings import warn

import geff
import numpy as np
import pandas as pd
import sqlalchemy as sqla
import zarr
from geff.core_io import construct_var_len_props, write_arrays
from geff_spec import Axis, PropMetadata
from sqlalchemy.orm import Session

from ultrack.config import MainConfig
from ultrack.core.database import NO_PARENT, LinkDB, NodeDB, OverlapDB
from ultrack.core.export.utils import solution_dataframe_from_sql
from ultrack.tracks.graph import add_track_ids_to_tracks_df

OVERLAPS_PATH = "overlaps/ids"

PROP_DESCRIPTIONS = {
    "track_id": (
        "Tracklet id, identical to the `track_id` of `to_tracks_layer` and to the "
        "label painted by `to_zarr`/`tracks_to_zarr`."
    ),
    "parent_track_id": "Track id of the parent tracklet, -1 for root tracklets.",
    "lineage_id": "Track id of the root tracklet of the lineage.",
    "seg_id": (
        "Label value of this node in the segmentation exported by "
        "`to_zarr`/`tracks_to_zarr` (equal to `track_id`)."
    ),
    "parent_id": "Ultrack node id of the parent node, -1 for roots.",
    "mask": "Binary mask of the node, cropped to `bbox`.",
    "bbox": "Bounding box of the node as (min (z), y, x, max (z), y, x) in pixels.",
}


# Helper function to convert pandas/numpy dtypes to string dtype names
def dtype_to_str(dtype) -> str:
    """Convert pandas/numpy dtype to string dtype name for PropMetadata."""
    # Convert to numpy dtype first to get consistent .name attribute
    np_dtype = np.dtype(dtype)
    dtype_name = np_dtype.name

    # Most dtypes work directly (int64, float64, bool, etc.)
    return dtype_name


def _to_sqlalchemy_url(database_path: Union[str, Path]) -> str:
    """Converts a database path to a SQLAlchemy URL, assuming SQLite for file paths."""
    database_path_str = str(database_path)
    if not database_path_str.startswith(
        ("sqlite://", "postgresql://", "mysql://", "postgresql+psycopg2://")
    ):
        database_path_str = f"sqlite:///{Path(database_path).absolute()}"
    return database_path_str


def _track_ids_dataframe(database_path: str) -> pd.DataFrame:
    """Computes `track_id`, `parent_track_id` and `lineage_id` of the solution nodes.

    It uses the same code path as `to_tracks_layer` so the track ids are identical
    to the ones from `to_tracks_layer` and the labels painted by `tracks_to_zarr`.

    Parameters
    ----------
    database_path : str
        SQLAlchemy database URL.

    Returns
    -------
    pd.DataFrame
        Dataframe indexed by node id with `track_id`, `parent_track_id` and
        `lineage_id` columns.
    """
    df = solution_dataframe_from_sql(database_path)
    df = add_track_ids_to_tracks_df(df)

    track_parent = (
        df[["track_id", "parent_track_id"]]
        .drop_duplicates("track_id")
        .set_index("track_id")["parent_track_id"]
        .to_dict()
    )

    lineage: Dict[int, int] = {}
    for track_id in track_parent:
        path = []
        current = track_id
        while current not in lineage and track_parent[current] != NO_PARENT:
            path.append(current)
            current = track_parent[current]
        root = lineage.get(current, current)
        lineage[current] = root
        for t in path:
            lineage[t] = root

    df["lineage_id"] = df["track_id"].map(lineage)

    return df[["track_id", "parent_track_id", "lineage_id"]].astype(np.int64)


def _make_axes(
    spatial_axes: Sequence[str],
    scale: Optional[Sequence[float]],
    spatial_unit: Optional[str],
    time_scale: Optional[float],
    time_unit: Optional[str],
) -> List[Axis]:
    """Creates the geff axes metadata.

    Coordinates are stored in pixels / frames, as ultrack stores them,
    `scale` and `scaled_unit` describe their conversion to physical units.
    """
    if time_unit is not None and time_scale is None:
        raise ValueError("`time_scale` must be provided when `time_unit` is set.")

    if spatial_unit is not None and scale is None:
        raise ValueError("`scale` must be provided when `spatial_unit` is set.")

    if scale is not None and len(scale) != len(spatial_axes):
        raise ValueError(
            f"`scale` must have one value per spatial axis {tuple(spatial_axes)}. "
            f"Got {tuple(scale)}."
        )

    axes = [
        Axis(
            name="t",
            type="time",
            unit="frame",
            scale=None if time_scale is None else float(time_scale),
            scaled_unit=time_unit,
        )
    ]
    for i, name in enumerate(spatial_axes):
        axes.append(
            Axis(
                name=name,
                type="space",
                unit="pixel",
                scale=None if scale is None else float(scale[i]),
                scaled_unit=spatial_unit,
            )
        )
    return axes


def to_geff_from_database(
    database_path: Union[str, Path],
    filename: Union[str, Path],
    overwrite: bool = False,
    *,
    zarr_format: Literal[2, 3] = 2,
    include_masks: bool = True,
    include_overlaps: bool = True,
    segmentation_path: Optional[str] = None,
    scale: Optional[Sequence[float]] = None,
    spatial_unit: Optional[str] = None,
    time_scale: Optional[float] = None,
    time_unit: Optional[str] = None,
) -> None:
    """
    Export tracks to a geff (Graph Exchange File Format) file from a database.

    Only the nodes and edges of the solution are exported.

    Besides ultrack's node attributes, each node stores:

    - `track_id`: tracklet id, identical to the `track_id` from `to_tracks_layer`
      and to the label painted by `to_zarr` / `tracks_to_zarr`.
    - `parent_track_id`: track id of the parent tracklet, -1 for roots.
    - `lineage_id`: track id of the lineage root tracklet.
    - `seg_id`: equal to `track_id`, the label of the node in the segmentation.

    `track_id` and `lineage_id` are declared as the geff `tracklet` and `lineage`
    `track_node_props`.

    The axes are `t, z, y, x` for 3D data and `t, y, x` for 2D data. Coordinates are
    stored in pixels (`unit="pixel"`) and frames (`unit="frame"`); the optional
    `scale` / `time_scale` and `spatial_unit` / `time_unit` are written as each axis'
    `scale` and `scaled_unit` to convert them into physical units.

    Parameters
    ----------
    database_path : str or Path
        The path to the database file.
    filename : str or Path
        The name of the file to save the tracks to.
    overwrite : bool, optional
        Whether to overwrite the file if it already exists, by default False.
    zarr_format : {2, 3}, optional
        Zarr format of the geff store, by default 2.
        Use 3 to store the geff inside an OME-Zarr v0.5 (zarr v3) store.
    include_masks : bool, optional
        Whether to store each node's binary mask as a boolean variable-length
        property, by default True.
    include_overlaps : bool, optional
        Whether to store the segmentation hypotheses overlaps as the custom
        `overlaps/ids` array, described in the metadata `extra["ultrack"]`,
        by default True.
    segmentation_path : str, optional
        Path of the segmentation labels exported by `to_zarr` / `tracks_to_zarr`,
        relative to the geff group (e.g. "../labels"). When provided, it is recorded
        as a `labels` related object linked through the `seg_id` node property.
    scale : Sequence[float], optional
        Physical size of a pixel for each spatial axis, (z, y, x) for 3D data and
        (y, x) for 2D data.
    spatial_unit : str, optional
        Physical unit of `scale` (e.g. "micrometer"). Requires `scale`.
    time_scale : float, optional
        Physical time interval between frames.
    time_unit : str, optional
        Physical unit of `time_scale` (e.g. "minute"). Requires `time_scale`.

    Raises
    ------
    FileExistsError
        If the file already exists and overwrite is False.
    ValueError
        If the solution is empty or the scale / unit parameters are inconsistent.
    """
    filename = Path(filename)
    if filename.exists():
        if not overwrite:
            raise FileExistsError(
                f"File {filename} already exists. Set `overwrite=True` to overwrite the file"
            )
        else:
            shutil.rmtree(filename)

    database_path_str = _to_sqlalchemy_url(database_path)
    engine = sqla.create_engine(database_path_str)
    with Session(engine) as session:
        # Collect nodes data, storing masks and bboxes separately
        all_nodes_data = []
        all_masks = []
        all_bboxes = []
        solution_source = []
        solution_target = []

        for (
            node_id,
            t,
            parent_id,
            z,
            y,
            x,
            z_shift,
            y_shift,
            x_shift,
            area,
            frontier,
            height,
            selected,
            pickle_obj,
        ) in session.query(
            NodeDB.id,
            NodeDB.t,
            NodeDB.parent_id,
            NodeDB.z,
            NodeDB.y,
            NodeDB.x,
            NodeDB.z_shift,
            NodeDB.y_shift,
            NodeDB.x_shift,
            NodeDB.area,
            NodeDB.frontier,
            NodeDB.height,
            NodeDB.selected,
            NodeDB.pickle,
        ).where(
            NodeDB.selected
        ):
            node_dict = {
                "id": node_id,
                "parent_id": parent_id,
                "t": t,
                "z": z,
                "y": y,
                "x": x,
                "z_shift": z_shift,
                "y_shift": y_shift,
                "x_shift": x_shift,
                "area": area,
                "frontier": frontier,
                "height": height,
                "solution": selected,
            }
            all_nodes_data.append(node_dict)
            all_bboxes.append(pickle_obj.bbox.astype(np.int64))
            if include_masks:
                all_masks.append(pickle_obj.mask.astype(bool))

            # Collect solution edges (parent-child relationships)
            if parent_id != NO_PARENT:
                solution_source.append(parent_id)
                solution_target.append(node_id)

        if len(all_nodes_data) == 0:
            raise ValueError(
                "No solution nodes found in the database. Run `solve` before exporting."
            )

        # Create nodes dataframe (only scalar values, no pickle objects)
        node_df = pd.DataFrame(all_nodes_data)
        node_df.set_index("id", inplace=True)
        node_df["solution"] = node_df["solution"].astype(bool)

        # Query edges
        edge_stmt = session.query(
            LinkDB.source_id, LinkDB.target_id, LinkDB.weight
        ).statement
        edge_df = pd.read_sql(edge_stmt, session.bind)

        # Add solution column to edges
        sol_links_df = pd.DataFrame(
            {
                "source_id": solution_source,
                "target_id": solution_target,
                "solution": True,
            }
        )
        edge_df = sol_links_df.merge(edge_df, on=["source_id", "target_id"], how="left")
        edge_df["solution"] = edge_df["solution"].fillna(True).astype(bool)
        if "weight" in edge_df.columns:
            edge_df["weight"] = edge_df["weight"].fillna(0.0).astype(np.float64)

        valid_edges = edge_df["target_id"].isin(node_df.index) & edge_df[
            "source_id"
        ].isin(node_df.index)
        if not valid_edges.all():
            invalid_edges = edge_df.loc[~valid_edges]
            edge_df = edge_df.loc[valid_edges]
            warn(f"Found invalid edges. They are being removed.\n{invalid_edges}")

        if include_overlaps:
            overlap_stmt = session.query(
                OverlapDB.node_id,
                OverlapDB.ancestor_id,
            ).statement
            overlap_df = pd.read_sql(overlap_stmt, session.bind)

    # bbox has (min, max) per spatial axis
    spatial_ndim = len(all_bboxes[0]) // 2
    if spatial_ndim == 3:
        spatial_axes = ["z", "y", "x"]
    elif spatial_ndim == 2:
        spatial_axes = ["y", "x"]
        node_df = node_df.drop(columns=["z", "z_shift"])
    else:
        raise ValueError(
            f"Expected 2D or 3D segments. Found bounding box {all_bboxes[0]}."
        )

    axes = _make_axes(spatial_axes, scale, spatial_unit, time_scale, time_unit)

    # Tracks information, computed with the same code path as `to_tracks_layer`
    tracks_df = _track_ids_dataframe(database_path_str)
    node_df = node_df.join(tracks_df, how="left")
    if node_df["track_id"].isna().any():
        raise RuntimeError("Some solution nodes were not assigned a `track_id`.")
    for c in tracks_df.columns:
        node_df[c] = node_df[c].astype(np.int64)
    node_df["seg_id"] = node_df["track_id"]

    # Create node properties metadata
    node_props_metadata = {}
    for c in node_df.columns:
        node_props_metadata[c] = PropMetadata(
            identifier=c,
            dtype=dtype_to_str(node_df[c].dtype),
            description=PROP_DESCRIPTIONS.get(c),
        )
    node_props_metadata["bbox"] = PropMetadata(
        identifier="bbox",
        dtype="int64",
        unit="pixel",
        description=PROP_DESCRIPTIONS["bbox"],
    )
    if include_masks:
        node_props_metadata["mask"] = PropMetadata(
            identifier="mask",
            dtype="bool",
            varlength=True,
            description=PROP_DESCRIPTIONS["mask"],
        )

    # Prepare edge IDs and properties
    edge_ids = np.column_stack(
        [
            edge_df["source_id"].to_numpy(dtype=np.uint64),
            edge_df["target_id"].to_numpy(dtype=np.uint64),
        ]
    ).reshape(-1, 2)
    edge_df = edge_df.drop(columns=["source_id", "target_id"])

    # Create edge properties metadata
    edge_props_metadata = {}
    for c in edge_df.columns:
        edge_props_metadata[c] = PropMetadata(
            identifier=c, dtype=dtype_to_str(edge_df[c].dtype)
        )

    related_objects = None
    if segmentation_path is not None:
        related_objects = [
            {"type": "labels", "path": str(segmentation_path), "node_prop": "seg_id"}
        ]

    extra = {}
    if include_overlaps:
        extra["ultrack"] = {
            "overlaps": {
                "path": OVERLAPS_PATH,
                "columns": ["ancestor_id", "node_id"],
                "description": (
                    "Pairs of overlapping segmentation hypotheses from ultrack's "
                    "hierarchical segmentation; ids may refer to non-selected "
                    "nodes that are not part of this graph."
                ),
            }
        }

    geff_metadata = geff.GeffMetadata(
        directed=True,
        axes=axes,
        node_props_metadata=node_props_metadata,
        edge_props_metadata=edge_props_metadata,
        track_node_props={"tracklet": "track_id", "lineage": "lineage_id"},
        related_objects=related_objects,
        extra=extra,
    )

    # Prepare node properties (using separately stored masks and bboxes)
    node_props = {}
    for c in node_df.columns:
        node_props[c] = {"values": node_df[c].to_numpy(), "missing": None}

    node_props["bbox"] = {"values": np.stack(all_bboxes), "missing": None}
    if include_masks:
        node_props["mask"] = construct_var_len_props(all_masks)

    edge_props = {}
    for c in edge_df.columns:
        edge_props[c] = {"values": edge_df[c].to_numpy(), "missing": None}

    write_arrays(
        filename,
        node_ids=node_df.index.to_numpy(dtype=np.uint64),
        node_props=node_props,
        edge_ids=edge_ids,
        edge_props=edge_props,
        metadata=geff_metadata,
        zarr_format=zarr_format,
    )

    if include_overlaps:
        # custom element to geff, described in metadata `extra["ultrack"]`
        store = zarr.open_group(filename, mode="a", zarr_format=zarr_format)
        store.create_array(
            OVERLAPS_PATH,
            data=overlap_df[["ancestor_id", "node_id"]]
            .to_numpy(dtype=np.uint64)
            .reshape(-1, 2),
        )
        store.create_group("overlaps/props")


def to_geff(
    config: MainConfig,
    filename: Union[str, Path],
    overwrite: bool = False,
    *,
    zarr_format: Literal[2, 3] = 2,
    include_masks: bool = True,
    include_overlaps: bool = True,
    segmentation_path: Optional[str] = None,
    scale: Optional[Sequence[float]] = None,
    spatial_unit: Optional[str] = None,
    time_scale: Optional[float] = None,
    time_unit: Optional[str] = None,
) -> None:
    """
    Export tracks to a geff (Graph Exchange File Format) file.

    See `to_geff_from_database` for the exported content.

    Parameters
    ----------
    config : MainConfig
        The configuration object.
    filename : str or Path
        The name of the file to save the tracks to.
    overwrite : bool, optional
        Whether to overwrite the file if it already exists, by default False.
    zarr_format : {2, 3}, optional
        Zarr format of the geff store, by default 2.
    include_masks : bool, optional
        Whether to store each node's binary mask, by default True.
    include_overlaps : bool, optional
        Whether to store the segmentation hypotheses overlaps, by default True.
    segmentation_path : str, optional
        Path of the segmentation labels exported by `to_zarr` / `tracks_to_zarr`,
        relative to the geff group, linked through the `seg_id` node property.
    scale : Sequence[float], optional
        Physical pixel size for each spatial axis, (z, y, x) or (y, x).
    spatial_unit : str, optional
        Physical unit of `scale` (e.g. "micrometer").
    time_scale : float, optional
        Physical time interval between frames.
    time_unit : str, optional
        Physical unit of `time_scale` (e.g. "minute").

    Raises
    ------
    FileExistsError
        If the file already exists and overwrite is False.
    """
    to_geff_from_database(
        database_path=config.data_config.database_path,
        filename=filename,
        overwrite=overwrite,
        zarr_format=zarr_format,
        include_masks=include_masks,
        include_overlaps=include_overlaps,
        segmentation_path=segmentation_path,
        scale=scale,
        spatial_unit=spatial_unit,
        time_scale=time_scale,
        time_unit=time_unit,
    )
