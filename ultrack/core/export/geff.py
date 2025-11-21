import shutil
from pathlib import Path
from typing import Union

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


# Helper function to convert pandas/numpy dtypes to string dtype names
def dtype_to_str(dtype) -> str:
    """Convert pandas/numpy dtype to string dtype name for PropMetadata."""
    # Convert to numpy dtype first to get consistent .name attribute
    np_dtype = np.dtype(dtype)
    dtype_name = np_dtype.name

    # Most dtypes work directly (int64, float64, bool, etc.)
    return dtype_name


def to_geff_from_database(
    database_path: Union[str, Path],
    filename: Union[str, Path],
    overwrite: bool = False,
) -> None:
    """
    Export tracks to a geff (Graph Exchange File Format) file from a database.

    Parameters
    ----------
    database_path : str or Path
        The path to the database file.
    filename : str or Path
        The name of the file to save the tracks to.
    overwrite : bool, optional
        Whether to overwrite the file if it already exists, by default False.

    Raises
    ------
    FileExistsError
        If the file already exists and overwrite is False.
    """
    filename = Path(filename)
    if filename.exists():
        if not overwrite:
            raise FileExistsError(
                f"File {filename} already exists. Set `overwrite=True` to overwrite the file"
            )
        else:
            shutil.rmtree(filename)

    # Convert database_path to SQLAlchemy URL format if needed
    database_path_str = str(database_path)
    # If it's not already a SQLAlchemy URL (doesn't start with a protocol), assume it's a SQLite file path
    if not database_path_str.startswith(
        ("sqlite://", "postgresql://", "mysql://", "postgresql+psycopg2://")
    ):
        # Convert file path to SQLite URL format
        database_path_str = f"sqlite:///{Path(database_path).absolute()}"
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
            # Store masks and bboxes separately
            all_masks.append(pickle_obj.mask.astype(np.uint64))
            all_bboxes.append(pickle_obj.bbox.astype(np.int64))

            # Collect solution edges (parent-child relationships)
            if selected and parent_id != NO_PARENT:
                solution_source.append(parent_id)
                solution_target.append(node_id)

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
        edge_df = edge_df.merge(sol_links_df, on=["source_id", "target_id"], how="left")
        edge_df.loc[edge_df["solution"].isna(), "solution"] = False
        edge_df["solution"] = edge_df["solution"].astype(bool)
        if "weight" in edge_df.columns:
            edge_df["weight"] = edge_df["weight"].astype(np.float64)

        # Query overlaps
        overlap_stmt = session.query(
            OverlapDB.node_id,
            OverlapDB.ancestor_id,
        ).statement
        overlap_df = pd.read_sql(overlap_stmt, session.bind)

    # Create node properties metadata
    node_props_metadata = {}
    for c in node_df.columns:
        node_props_metadata[c] = PropMetadata(
            identifier=c,
            dtype=dtype_to_str(node_df[c].dtype),
        )
    node_props_metadata["mask"] = PropMetadata(
        identifier="mask",
        dtype="uint64",
        varlength=True,
    )
    node_props_metadata["bbox"] = PropMetadata(
        identifier="bbox",
        dtype="int64",
    )

    # Prepare edge IDs and properties
    edge_ids = np.column_stack(
        [
            edge_df["source_id"].to_numpy(dtype=np.uint64),
            edge_df["target_id"].to_numpy(dtype=np.uint64),
        ]
    )
    edge_df = edge_df.drop(columns=["source_id", "target_id"])

    # Create edge properties metadata
    edge_props_metadata = {}
    for c in edge_df.columns:
        edge_props_metadata[c] = PropMetadata(
            identifier=c, dtype=dtype_to_str(edge_df[c].dtype)
        )

    geff_metadata = geff.GeffMetadata(
        directed=True,
        axes=[
            Axis(name="t", type="time"),
            Axis(name="z", type="space"),
            Axis(name="y", type="space"),
            Axis(name="x", type="space"),
        ],
        node_props_metadata=node_props_metadata,
        edge_props_metadata=edge_props_metadata,
    )

    # Prepare node properties (using separately stored masks and bboxes)
    node_props = {}
    for c in node_df.columns:
        # Convert to appropriate numpy dtype
        values = node_df[c].to_numpy()
        node_props[c] = {"values": values, "missing": None}

    # Handle mask - use the separately stored masks
    node_props["mask"] = construct_var_len_props(all_masks)

    # Handle bbox - stack into 2D array from separately stored bboxes
    bbox_array = np.stack(all_bboxes)
    node_props["bbox"] = {"values": bbox_array, "missing": None}

    # Prepare edge properties with proper dtypes
    edge_props = {}
    for c in edge_df.columns:
        values = edge_df[c].to_numpy()
        edge_props[c] = {"values": values, "missing": None}

    write_arrays(
        filename,
        node_ids=node_df.index.to_numpy(dtype=np.uint64),
        node_props=node_props,
        edge_ids=edge_ids,
        edge_props=edge_props,
        metadata=geff_metadata,
    )

    # custom element to geff
    store = zarr.open(filename, mode="a")
    store["overlaps/ids"] = overlap_df[["ancestor_id", "node_id"]].to_numpy(
        dtype=np.uint64
    )
    store.create_group("overlaps/props")


def to_geff(
    config: MainConfig,
    filename: Union[str, Path],
    overwrite: bool = False,
) -> None:
    """
    Export tracks to a geff (Graph Exchange File Format) file.

    Parameters
    ----------
    config : MainConfig
        The configuration object.
    filename : str or Path
        The name of the file to save the tracks to.
    overwrite : bool, optional
        Whether to overwrite the file if it already exists, by default False.

    Raises
    ------
    FileExistsError
        If the file already exists and overwrite is False.
    """
    to_geff_from_database(
        database_path=config.data_config.database_path,
        filename=filename,
        overwrite=overwrite,
    )
