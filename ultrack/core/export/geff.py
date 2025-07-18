from pathlib import Path
from typing import Union

import geff
import networkx as nx

from ultrack.config import MainConfig
from ultrack.core.export.networkx import to_networkx


def to_geff(
    config: MainConfig,
    filename: Union[str, Path],
    overwrite: bool = False,
) -> None:
    """
    Export tracks to a geff (Graph Exchange File Format) file.

    Parametersmnist
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
    if Path(filename).exists() and not overwrite:
        raise FileExistsError(
            f"File {filename} already exists. Set `overwrite=True` to overwrite the file"
        )

    # Get the networkx graph from the configuration
    graph = to_networkx(config)

    # Write the graph to geff format
    geff.write_nx(graph, filename)
