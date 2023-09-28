import logging
from pathlib import Path
from typing import Callable, List, Union

import pandas as pd
from napari.types import LayerDataTuple

from ultrack.tracks.graph import inv_tracks_df_forest

LOG = logging.getLogger(__name__)

TRACKS_HEADER = ("track_id", "t", "z", "y", "x")


def napari_get_reader(
    path: Union[str, List[str]]
) -> Callable[[Union[str, List[str]]], List[LayerDataTuple]]:
    """
    Return a Napari reader function for CSV tracks data.

    Parameters
    ----------
    path : Union[str, List[str]]
        Path or list of paths to the CSV file(s).

    Returns
    -------
    Callable[[Union[str, List[str]]], List[LayerDataTuple]]
        A callable Napari reader function.

    Notes
    -----
    This function returns a Napari reader function that can be used to read CSV
    files containing track data into Napari.

    If the input is a list of paths, only the first path will be considered.

    If the path ends with '.csv' and exists, the reader function is returned,
    otherwise None is returned.
    """
    if isinstance(path, list):
        path = path[0]

    if isinstance(path, str):
        path = Path(path)

    LOG.info(f"Reading tracks from {path}")

    if not path.name.endswith(".csv"):
        LOG.info(f"{path} must end with `.csv`.")
        return None

    if not path.exists():
        LOG.info(f"{path} does not exist.")
        return None

    header = pd.read_csv(path, nrows=0).columns.tolist()
    LOG.info(f"Tracks file header: {header}")

    for colname in TRACKS_HEADER:
        if colname == "z":
            continue

        if colname not in header:
            LOG.info(f"{path} must contain `{colname}` column.")
            return None

    return reader_function


def read_csv(path: Union[Path, str]) -> LayerDataTuple:
    """
    Read track data from a CSV file.

    Parameters
    ----------
    path : Union[Path, str]
        Path to the CSV file.

    Returns
    -------
    LayerDataTuple
        A tuple containing the track data, metadata, and layer type.

    Notes
    -----
    This function reads track data from a CSV file and returns it in a format
    suitable for display in Napari.

    If the CSV file contains a 'parent_track_id' column, a track lineage graph
    is constructed.
    """
    if isinstance(path, str):
        path = Path(path)

    df = pd.read_csv(path)

    LOG.info(f"Read {len(df)} tracks from {path}")
    LOG.info(df.head())

    tracks_cols = list(TRACKS_HEADER)
    if "z" not in df.columns:
        tracks_cols.remove("z")

    if "parent_track_id" in df.columns:
        graph = inv_tracks_df_forest(df)
        LOG.info(f"Track lineage graph with length {len(graph)}")
    else:
        graph = None

    kwargs = {
        "features": df,
        "name": path.name.removesuffix(".csv"),
        "graph": graph,
    }

    return (df[tracks_cols], kwargs, "tracks")


def reader_function(path: Union[List[str], str]) -> List:
    """
    Dispatch single files for the `read_csv` function.

    Parameters
    ----------
    path : Union[List[str], str]
        Path or list of paths to the CSV file(s).

    Returns
    -------
    List
        List of track data tuples.
    """
    paths = [path] if isinstance(path, (str, Path)) else path
    return [read_csv(p) for p in paths]
