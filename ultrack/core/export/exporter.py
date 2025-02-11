import json
from pathlib import Path
from typing import Union

import networkx as nx

from ultrack.config import MainConfig
from ultrack.core.export import (
    to_networkx,
    to_trackmate,
    to_tracks_layer,
    tracks_to_zarr,
)


def export_tracks_by_extension(
    config: MainConfig, filename: Union[str, Path], overwrite: bool = False
) -> None:
    """
    Export tracks to a file given the file extension.

    Supported file extensions are .xml, .csv, .zarr, .dot, and .json.
    - `.xml` exports to a TrackMate compatible XML file.
    - `.csv` exports to a CSV file.
    - `.zarr` exports the tracks to dense segments in a `zarr` array format.
    - `.dot` exports to a Graphviz DOT file.
    - `.json` exports to a networkx JSON file.

    Parameters
    ----------
    filename : str or Path
        The name of the file to save the tracks to.
    config : MainConfig
        The configuration object.
    overwrite : bool, optional
        Whether to overwrite the file if it already exists, by default False.

    See Also
    --------
    to_trackmate :
        Export tracks to a TrackMate compatible XML file.
    to_tracks_layer :
        Export tracks to a CSV file.
    tracks_to_zarr :
        Export tracks to a `zarr` array.
    to_networkx :
        Export tracks to a networkx graph.
    """
    if Path(filename).exists() and not overwrite:
        raise FileExistsError(
            f"File {filename} already exists. Set `overwrite=True` to overwrite the file"
        )

    file_ext = Path(filename).suffix
    if file_ext.lower() == ".xml":
        to_trackmate(config, filename, overwrite=True)
    elif file_ext.lower() == ".csv":
        df, _ = to_tracks_layer(config, include_parents=True)
        df.to_csv(filename, index=False)
    elif file_ext.lower() == ".zarr":
        df, _ = to_tracks_layer(config)
        tracks_to_zarr(config, df, filename, overwrite=True)
    elif file_ext.lower() == ".dot":
        G = to_networkx(config)
        nx.drawing.nx_pydot.write_dot(G, filename)
    elif file_ext.lower() == ".json":
        G = to_networkx(config)
        json_data = nx.node_link_data(G)
        with open(filename, "w") as f:
            json.dump(json_data, f)
    else:
        raise ValueError(
            f"Unknown file extension: {file_ext}. Supported extensions are .xml, .csv, .zarr, .dot, and .json."
        )
