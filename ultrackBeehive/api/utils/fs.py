import uuid
from pathlib import Path
from typing import Union

import ome_zarr
from fastapi import HTTPException
from ome_zarr.io import parse_url
from ome_zarr.reader import Reader


def create_temp_dir(path: Path, prefix: str = "", ext: str = "") -> Path:
    """Create a temporary directory.

    Parameters
    ----------
    path : Path
        The path to the directory where the temporary dir will be created.
    prefix : str
        The prefix of the temporary directory name. Defaults to "".
    ext : str
        The extension of the temporary directory name. Not required, but it is useful to
        identify zarr arrays. Defaults to "".

    Raises
    ------
    ValueError
        If the temporary directory could not be created.

    Returns
    -------
    Path
        The path to the created temporary directory
    """
    temp_dir = path / f"{prefix}{uuid.uuid4().hex}.{ext}"
    temp_dir.mkdir(exist_ok=True, parents=True)
    if not temp_dir.is_dir():
        raise ValueError(f"Error creating temporary directory {temp_dir}.")
    return temp_dir


def create_temp_zarr_path(*args, **kwargs) -> Path:
    """Create a temporary zarr directory.

    It is a wrapper around `create_temp_dir` with `ext="zarr"`.

    See Also
    --------
    create_temp_dir: original wrapped function.
    """
    return create_temp_dir(*args, ext="zarr", **kwargs)


def open_image(path: Union[str, Path]) -> ome_zarr.reader.Node:
    """Open an ome-zarr image.

    Parameters
    ----------
    path : Union[str, Path]
        The path to the ome-zarr image.

    Raises
    ------
    HTTPException
        If the image could not be opened.

    ValueError
        If the image is not a single image.

    Returns
    -------
    ome_zarr.reader.Node
        The opened image.
    """
    try:
        reader = Reader(parse_url(path))
        nodes = list(reader())
        if len(nodes) != 1:
            raise ValueError(f"Expected a single image, found {len(nodes)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Could not open {path}. {str(e)}")
    return nodes[0]
