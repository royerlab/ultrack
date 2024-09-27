from typing import Dict, List, Tuple, Union

import ome_zarr
import zarr


def _get_omero_spec(node: ome_zarr.reader.Node) -> Dict:
    """Get the OMERO spec from the ome-zarr image.

    Parameters
    ----------
    node : ome_zarr.reader.Node
        The ome-zarr image node.

    Raises
    ------
    ValueError
        If the image doesn't have an OMERO spec.
    """
    for spec in node.specs:
        if isinstance(spec, ome_zarr.reader.OMERO):
            return spec.image_data["channels"]
    raise ValueError("No OMERO spec found.")


def get_channels_from_ome_zarr(
    node: ome_zarr.reader.Node, valid_channels: List[Tuple[str, str]]
) -> Dict[str, Dict[str, Union[zarr.Array, dict]]]:
    """Get the data of the specified channels from the ome-zarr image.

    Parameters
    ----------
    node : ome_zarr.reader.Node
        The ome-zarr image node.
    valid_channels : List[Tuple[str, str]]
        List of tuples with the name of the channel and the channel name in the image.

    Raises
    ------
    ValueError
        If the image doesn't have any of the specified channels.

    Returns
    -------
    Dict[str, Dict[str, zarr.Array]]
        Dictionary with the data of the specified channels.

    Examples
    --------
    >>> import ultrack.api.utils as api_utils
    >>> api_utils.fs.open_image("https://public.czbiohub.org/royerlab/zebrahub/imaging/single-objective/ZSNS001.ome.zarr") # noqa
    >>> named_data = api_utils.zarr.get_channels_from_ome_zarr(node, [("image", "fused")])
    >>> zarr_data = named_data["image"]["data"]
    """
    valid_channels = {
        name: channel for name, channel in valid_channels if channel is not None
    }

    if len(valid_channels) == 0:
        raise ValueError(f"node {node} doesn't have any channels.")

    channel_dim_index = -1

    for i, axis in enumerate(node.metadata["axes"]):
        if axis["type"] == "channel":
            channel_dim_index = i
            break

    if channel_dim_index == -1:
        raise ValueError("The image doesn't have a channel axis.")

    named_data = {}
    for name, channel in valid_channels.items():
        try:
            image_index = node.metadata["channel_names"].index(channel)
        except ValueError as ex:
            raise ValueError(
                f"node {node} doesn't have a channel named {channel}. {ex}"
            )

        channel_slice = [
            slice(None) if axis_idx != channel_dim_index else image_index
            for axis_idx in range(node.data[0].ndim)
        ]
        image_data = node.data[0][tuple(channel_slice)]
        named_data[name] = image_data

    return named_data
