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
from geff_spec import Axis, PropMetadata, RelatedObject
from geff_spec.utils import axes_from_lists, create_props_metadata
from sqlalchemy.orm import Session

from ultrack.config import MainConfig
from ultrack.core.database import NO_PARENT, LinkDB, NodeDB, OverlapDB
from ultrack.tracks.graph import (
    add_track_ids_to_tracks_df,
    get_subtree,
    tracks_df_forest,
)

OVERLAPS_PATH = "overlaps/ids"

PROP_UNITS = {
    "t": "frame",
    "z": "pixel",
    "y": "pixel",
    "x": "pixel",
    "z_shift": "pixel",
    "y_shift": "pixel",
    "x_shift": "pixel",
    "bbox": "pixel",
}

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


def _to_sqlalchemy_url(database_path: Union[str, Path]) -> str:
    """Converts a database path to a SQLAlchemy URL, assuming SQLite for file paths."""
    database_path_str = str(database_path)
    if not database_path_str.startswith(
        ("sqlite://", "postgresql://", "mysql://", "postgresql+psycopg2://")
    ):
        database_path_str = f"sqlite:///{Path(database_path).absolute()}"
    return database_path_str


def _axes(
    spatial_axes: Sequence[str],
    scale: Optional[Sequence[float]],
    spatial_unit: Optional[str],
    time_scale: Optional[float],
    time_unit: Optional[str],
) -> List[Axis]:
    """Creates the geff axes, `t` followed by `spatial_axes`.

    Coordinates are stored in frames / pixels, as ultrack stores them, `scale` and
    `scaled_unit` describe their conversion to physical units.
    """
    n_spatial = len(spatial_axes)

    if scale is not None:
        if len(scale) == n_spatial + 1:
            # (t, (z), y, x) scale, as stored by the CLI in the data config metadata
            scale = scale[1:]
        elif len(scale) != n_spatial:
            raise ValueError(
                f"`scale` must have one value per spatial axis {tuple(spatial_axes)}. "
                f"Got {tuple(scale)}."
            )
        scale = [float(s) for s in scale]
    else:
        scale = [None] * n_spatial

    return axes_from_lists(
        axis_names=["t", *spatial_axes],
        axis_types=["time"] + ["space"] * n_spatial,
        axis_units=["frame"] + ["pixel"] * n_spatial,
        axis_scales=[None if time_scale is None else float(time_scale), *scale],
        scaled_units=[time_unit] + [spatial_unit] * n_spatial,
    )


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
        relative to the geff group's zarr attributes (e.g. "../labels").
        When provided, it is recorded as a `labels` related object linked through
        the `seg_id` node property.
    scale : Sequence[float], optional
        Physical size of a pixel for each spatial axis, (z, y, x) for 3D data and
        (y, x) for 2D data. A leading time entry, as stored by the CLI in the data
        config metadata, is ignored.
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

    axes = _axes(spatial_axes, scale, spatial_unit, time_scale, time_unit)

    # Tracks information, same code path as `to_tracks_layer` so the ids are identical
    add_track_ids_to_tracks_df(node_df)
    forest = tracks_df_forest(node_df)
    roots = forest.pop(NO_PARENT)
    lineage = {t: r for r in roots for t in get_subtree(forest, r)}
    node_df["lineage_id"] = node_df["track_id"].map(lineage).astype(np.int64)
    node_df["seg_id"] = node_df["track_id"]

    node_props = {
        c: {"values": node_df[c].to_numpy(), "missing": None} for c in node_df.columns
    }
    node_props["bbox"] = {"values": np.stack(all_bboxes), "missing": None}
    if include_masks:
        node_props["mask"] = construct_var_len_props(all_masks)

    node_props_metadata: Dict[str, PropMetadata] = {
        name: create_props_metadata(
            name,
            prop,
            unit=PROP_UNITS.get(name),
            description=PROP_DESCRIPTIONS.get(name),
        )
        for name, prop in node_props.items()
    }

    edge_ids = np.column_stack(
        [
            edge_df.pop("source_id").to_numpy(dtype=np.uint64),
            edge_df.pop("target_id").to_numpy(dtype=np.uint64),
        ]
    )
    edge_props = {
        c: {"values": edge_df[c].to_numpy(), "missing": None} for c in edge_df.columns
    }

    related_objects = None
    if segmentation_path is not None:
        related_objects = [
            RelatedObject(
                type="labels", path=str(segmentation_path), node_prop="seg_id"
            )
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
        edge_props_metadata={},  # inferred from `edge_props` by `write_arrays`
        track_node_props={"tracklet": "track_id", "lineage": "lineage_id"},
        related_objects=related_objects,
        extra=extra,
    )

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
        group = zarr.open_group(filename, mode="a", zarr_format=zarr_format)
        group[OVERLAPS_PATH] = overlap_df[["ancestor_id", "node_id"]].to_numpy(
            dtype=np.uint64
        )
        group.create_group("overlaps/props")


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
        relative to the geff group's zarr attributes, linked through the `seg_id`
        node property.
    scale : Sequence[float], optional
        Physical pixel size for each spatial axis, (z, y, x) or (y, x).
        By default, the `scale` of the data config metadata, if present.
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
    if scale is None:
        scale = config.data_config.metadata.get("scale")

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
