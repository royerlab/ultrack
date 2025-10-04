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
    filename = Path(filename)
    if filename.exists():
        if not overwrite:
            raise FileExistsError(
                f"File {filename} already exists. Set `overwrite=True` to overwrite the file"
            )
        else:
            shutil.rmtree(filename)

    engine = sqla.create_engine(config.data_config.database_path)
    with Session(engine) as session:
        node_stmt = session.query(
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
        ).statement
        node_df = pd.read_sql(node_stmt, session.bind, index_col="id")
        node_df["id"] = node_df.index

        edge_stmt = session.query(
            LinkDB.source_id, LinkDB.target_id, LinkDB.weight
        ).statement
        edge_df = pd.read_sql(edge_stmt, session.bind)

        sol_links_df = node_df.loc[
            node_df["selected"] & node_df["parent_id"] != NO_PARENT,
            ["id", "parent_id"],
        ]
        sol_links_df = sol_links_df.rename(
            columns={"parent_id": "source_id", "id": "target_id"},
        )
        sol_links_df["solution"] = True
        edge_df = edge_df.merge(sol_links_df, on=["source_id", "target_id"])
        edge_df["solution"] = edge_df["solution"].fillna(False)

        node_df.rename(columns={"selected": "solution"}, inplace=True)
        node_df.drop(["id", "parent_id"], axis=1, inplace=True)

        overlap_stmt = session.query(
            OverlapDB.node_id,
            OverlapDB.ancestor_id,
        ).statement
        overlap_df = pd.read_sql(overlap_stmt, session.bind)

    node_props_metadata = {
        c: PropMetadata(
            identifier=c,
            dtype=node_df[c].dtype,
        )
        for c in node_df.columns
        if c != "pickle"
    }
    node_props_metadata["mask"] = PropMetadata(
        identifier="mask",
        dtype=np.uint64,
        varlength=True,
    )
    node_props_metadata["bbox"] = PropMetadata(
        identifier="bbox",
        dtype=np.int64,
    )

    edge_ids = edge_df[["source_id", "target_id"]].to_numpy(dtype=np.uint64)
    edge_df = edge_df.drop(columns=["source_id", "target_id"])

    edge_props_metadata = {
        c: PropMetadata(identifier=c, dtype=edge_df[c].dtype) for c in edge_df.columns
    }

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

    node_props = {
        c: {"values": node_df[c].to_numpy(), "missing": None}
        for c in node_df.columns
        if c != "pickle"
    }
    node_props["mask"] = construct_var_len_props(
        [v.mask.astype(np.uint64) for v in node_df["pickle"]]
    )
    node_props["bbox"] = {
        "values": np.stack([v.bbox for v in node_df["pickle"]]),
        "missing": None,
    }

    write_arrays(
        filename,
        node_ids=node_df.index.to_numpy(dtype=np.uint64),
        node_props=node_props,
        edge_ids=edge_ids,
        edge_props={
            c: {"values": edge_df[c].to_numpy(), "missing": None} for c in edge_df
        },
        metadata=geff_metadata,
    )

    # custom element to geff
    store = zarr.open(filename, mode="a")
    store["overlaps/ids"] = overlap_df[["ancestor_id", "node_id"]].to_numpy(
        dtype=np.uint64
    )
    store.create_group("overlaps/props")
