import logging
import math as m
from pathlib import Path
from typing import Optional, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd
import sqlalchemy as sqla
import torch as th
import torch.nn.functional as F
import zarr
from numpy.typing import ArrayLike
from skimage.measure import regionprops_table
from sqlalchemy.orm import Session
from tqdm import tqdm
from zarr.storage import Store

from ultrack.config import MainConfig
from ultrack.core.database import NodeDB
from ultrack.utils.array import create_zarr, large_chunk_size
from ultrack.utils.constants import ULTRACK_DEBUG
from ultrack.utils.cuda import import_module, torch_default_device

logging.basicConfig()
logging.getLogger("sqlachemy.engine").setLevel(logging.INFO)

LOG = logging.getLogger(__name__)

try:
    import cupy as xp

    LOG.info("cupy found.")

except (ModuleNotFoundError, ImportError):
    import numpy as xp

    LOG.info("cupy not found using numpy and scipy.")


_ALIGN_CORNERS = True


def _interpolate(tensor: th.Tensor, antialias: bool = False, **kwargs) -> th.Tensor:
    """Interpolates tensor.

    Parameters
    ----------
    tensor : th.Tensor
        Input 4 or 5-dim tensor.
    antialias : bool, optional
        When true applies a gaussian filter with 0.5 * downscale factor.
        Ignored if upscaling.

    Returns
    -------
    th.Tensor
        Interpolated tensor.
    """
    mode = "trilinear" if tensor.ndim == 5 else "bilinear"
    if antialias:
        scale_factor = kwargs.get("scale_factor")
        if scale_factor is None:
            raise ValueError(
                "`_interpolate` with `antialias=True` requires `scale_factor` parameter."
            )

        if scale_factor < 1.0:
            ndi = import_module("scipy", "ndimage")
            orig_shape = tensor.shape

            if xp == np:  # xp ideally should be cupy in this case
                LOG.warning(
                    "Cupy not found, using scipy for interpolation. This may slow down ultrack."
                )
                array = xp.asarray(tensor.cpu().squeeze().contiguous().numpy())
            else:
                array = xp.asarray(tensor.squeeze().contiguous())
            blurred = ndi.gaussian_filter(
                array,
                sigma=0.5 / scale_factor,
                output=array,
            )
            tensor = th.as_tensor(blurred, device=tensor.device)
            tensor = tensor.reshape(orig_shape)
            LOG.info(f"Antialiasing with sigma = {0.5 / scale_factor}.")

    return F.interpolate(tensor, **kwargs, mode=mode, align_corners=_ALIGN_CORNERS)


def total_variation_loss(tensor: th.Tensor) -> th.Tensor:
    loss = 0.0
    for i in range(1, tensor.ndim - 1):
        idx = th.arange(tensor.shape[i], device=tensor.device)
        # tv = th.square(
        #     2 * th.index_select(tensor, i, idx[1:-1]) \
        #       - th.index_select(tensor, i, idx[2:]) \
        #       - th.index_select(tensor, i, idx[:-2])
        # )  # second derivative
        tv = th.square(
            th.index_select(tensor, i, idx[:-1]) - th.index_select(tensor, i, idx[1:])
        )  # first derivative

        loss = loss + tv.sum()
    return loss


def identity_grid(shape: tuple[int, ...]) -> th.Tensor:
    """Grid equivalent to a identity vector field (no flow).

    Parameters
    ----------
    shape : tuple[int, ...]
        Grid shape.

    Returns
    -------
    th.Tensor
        Tensor of shape (Z, Y, X, D)
    """
    ndim = len(shape)
    T = th.zeros((1, ndim, ndim + 1))
    T[:, :, :-1] = th.eye(ndim)
    grid_shape = (1, 1) + shape
    return F.affine_grid(T, grid_shape, align_corners=_ALIGN_CORNERS)


def _interpolate_grid(grid: th.Tensor, out_dim: int = -1, **kwargs) -> th.Tensor:
    """Interpolate a grid.

    Parameters
    ----------
    grid : th.Tensor
        Grid tensor of shape (z, y, x, D).
    out_dim : int, optional
        Output dimension, by default -1.

    Returns
    -------
    th.Tensor
        Tensor of shape (Z, Y, X, D) when out_dim=1, D position is subject to `out_dim` value.
    """
    return th.stack(
        [_interpolate(grid[None, ..., d], **kwargs)[0] for d in range(grid.shape[-1])],
        dim=out_dim,
    )


def flow_field(
    source: th.Tensor,
    target: th.Tensor,
    im_factor: int = 4,
    grid_factor: int = 4,
    num_iterations: int = 1000,
    lr: float = 1e-2,
    n_scales: int = 3,
    init_grid: Optional[th.Tensor] = None,
) -> th.Tensor:
    """
    Compute the flow vector field `T` that minimizes the
    mean squared error between `T(source)` and `target`.

    Parameters
    ----------
    source : torch.Tensor
        Source image (C, Z, Y, X).
    target : torch.Tensor
        Target image (C, Z, Y, X).
    im_factor : int, optional
        Image space down scaling factor, by default 4.
    grid_factor : int, optional
        Grid space down scaling factor, by default 4.
        Grid dimensions will be divided by both `im_factor` and `grid_factor`.
    num_iterations : int, optional
        Number of gradient descent iterations, by default 1000.
    lr : float, optional
        Learning rate (gradient descent step), by default 1e-2
    n_scales : int, optional
        Number of scales used for multi-scale optimization, by default 3.

    Returns
    -------
    torch.Tensor
        Vector field array with shape (D, (Z / factor), Y / factor, X / factor)
    """
    assert source.shape == target.shape
    assert n_scales > 0
    ndim = source.ndim - 1

    if init_grid is not None:
        if init_grid.ndim != ndim + 2:
            raise ValueError(
                f"Grid must have ndim dimensions ({ndim + 2}), (B, (Z), Y, X, D). Found {init_grid.ndim}."
            )

        if init_grid.shape[-1] != ndim:
            raise ValueError(
                f"Grid last channel must match input dimensions. Found {init_grid.shape[-1]} expected {ndim}."
            )

    source = source.unsqueeze(0)
    target = target.unsqueeze(0)

    device = source.device
    scales = np.flip(np.power(2, np.arange(n_scales)))
    grid = init_grid
    norm_factor = None
    constant = 1_000  # scale of input affects the results, 1_000 is a good value

    for scale in scales:
        with th.no_grad():
            scaled_im_factor = im_factor * scale
            scaled_source = _interpolate(
                source, scale_factor=1 / scaled_im_factor, antialias=True
            )
            scaled_target = _interpolate(
                target, scale_factor=1 / scaled_im_factor, antialias=True
            )
            if norm_factor is None:
                norm_factor = (
                    2.0
                    * constant
                    / (scaled_source.quantile(0.9999) + scaled_target.quantile(0.9999))
                )

            scaled_source *= norm_factor
            scaled_target *= norm_factor

            grid_shape = tuple(
                m.ceil(s / grid_factor) for s in scaled_source.shape[-ndim:]
            )

            if grid is None:
                grid = identity_grid(grid_shape).to(device)
                grid0 = identity_grid(grid_shape).to(device)
            else:
                grid = _interpolate_grid(grid, size=grid_shape)
                grid0 = identity_grid(grid_shape).to(device)

        grid.requires_grad_(True).retain_grad()

        LOG.info(f"scale: {scale}")
        LOG.info(f"image shape: {scaled_source.shape}")
        LOG.info(f"grid shape: {grid.shape}")

        for i in range(num_iterations):
            if grid.grad is not None:
                grid.grad.zero_()

            large_grid = _interpolate_grid(grid, size=scaled_source.shape[-ndim:])

            im2hat = F.grid_sample(
                scaled_target, large_grid, align_corners=_ALIGN_CORNERS
            )

            loss = F.l1_loss(im2hat, scaled_source)
            loss = loss + total_variation_loss(grid - grid0)
            loss.backward()

            if i % 10 == 0:
                LOG.info(f"iter. {i} MSE: {loss:0.4f}")

            with th.no_grad():
                grid -= lr * grid.grad

    LOG.info(f"image size: {source.shape}")
    LOG.info(f"image factor: {im_factor}")
    LOG.info(f"grid factor: {grid_factor}")

    with th.no_grad():
        grid = cast(th.Tensor, grid)
        grid = grid - grid0
        grid = th.flip(grid, (-1,))  # x, y, z -> z, y, x

        # divided by 2.0 because the range is -1 to 1 (length = 2.0)
        grid /= 2.0
        grid = _interpolate_grid(grid, out_dim=1, size=scaled_source.shape[-ndim:])[0]

        LOG.info(f"vector field shape: {grid.shape}")

    if ULTRACK_DEBUG:
        import napari

        viewer = napari.Viewer()
        viewer.add_image(
            scaled_source.cpu().numpy(),
            name="im1",
            blending="additive",
            colormap="blue",
            visible=False,
        )
        viewer.add_image(
            scaled_target.cpu().numpy(),
            name="im2",
            blending="additive",
            colormap="green",
        )
        viewer.add_image(
            im2hat.detach().cpu().numpy(),
            name="im2hat",
            blending="additive",
            colormap="red",
        )
        viewer.add_image(
            grid.detach().cpu().numpy(),
            name="grid",
            colormap="turbo",
            visible=False,
        )
        napari.run()

    return grid


@th.no_grad()
def apply_field(field: th.Tensor, image: th.Tensor) -> th.Tensor:
    """
    Transform image using vector field.
    Image will be scaled to the field size.

    Parameters
    ----------
    field : th.Tensor
        Vector field (D, z, y, x)
    image : th.Tensor
        Original image used to compute the vector field.

    Returns
    -------
    th.Tensor
        Transformed image (z, y, x)
    """
    assert image.ndim == 4

    field = th.flip(field, (0,))  # z, y, x -> x, y, z
    field = field.movedim(0, -1)[None]

    field = field * 2.0  # mapping range from image shape to -1 to 1
    field = identity_grid(field.shape[1:-1]).to(field.device) - field

    transformed_image = F.grid_sample(image[None], field, align_corners=_ALIGN_CORNERS)

    return transformed_image[0]


@th.no_grad()
def advenct_field(
    field: ArrayLike,
    sources: th.Tensor,
    shape: Optional[tuple[int, ...]] = None,
    invert: bool = True,
) -> th.Tensor:
    """
    Advenct points from sources through the provided field.
    Shape indicates the original shape (space) and sources.
    Useful when field is down scaled from the original space.

    Parameters
    ----------
    field : ArrayLike
        Field array with shape T x D x (Z) x Y x X
    sources : th.Tensor
        Array of sources N x D
    shape : tuple[int, ...]
        When provided scales field accordingly, D-dimensional tuple.
    invert : bool
        When true flow is multiplied by -1, resulting in reversal of the flow.

    Returns
    -------
    th.Tensor
        Trajectories of sources N x T x D
    """
    ndim = field.ndim - 2
    device = sources.device
    orig_shape = th.tensor(shape, device=device)
    field_shape = th.tensor(field.shape[2:], device=device)

    if orig_shape is None:
        scales = th.ones(ndim, device=device)
    else:
        scales = (field_shape - 1) / (orig_shape - 1)

    trajectories = [sources]

    zero = th.zeros(1, device=device)

    # ignore last (first) frame since it will be empty
    iterator = range(1, field.shape[0]) if invert else range(0, field_shape[0] - 1)

    for t in iterator:
        current = th.as_tensor(field[t]).to(device=device, non_blocking=True)

        int_sources = th.round(trajectories[-1] * scales)
        int_sources = th.maximum(int_sources, zero)
        int_sources = th.minimum(int_sources, field_shape - 1).long()
        spatial_idx = tuple(
            t.T[0] for t in th.tensor_split(int_sources, len(orig_shape), dim=1)
        )
        idx = (slice(None), *spatial_idx)

        movement = current[idx].T * orig_shape

        if invert:
            sources = sources - movement
        else:
            sources = sources + movement

        trajectories.append(sources)

    trajectories = th.stack(trajectories, dim=1)

    return trajectories


def advenct_field_from_labels(
    field: ArrayLike,
    label: ArrayLike,
    invert: bool = True,
) -> ArrayLike:
    """
    Advenct points from segmentation labels centroid.

    Parameters
    ----------
    field : ArrayLike
        Field array with shape T x D x (Z) x Y x X
    label : ArrayLike
        Label image.
    invert : bool
        When true flow is multiplied by -1, resulting in reversal of the flow.

    Returns
    -------
    ArrayLike
        Trajectories of sources N x T x D
    """
    if isinstance(label, th.Tensor):
        device = label.device
        label = label.numpy()
    else:
        device = torch_default_device()

    centroids = np.stack(
        [v for v in regionprops_table(label, properties=("centroid",)).values()], axis=1
    )
    centroids = th.as_tensor(centroids, device=device)

    trajectories = advenct_field(field, centroids, shape=label.shape, invert=invert)

    trajectories = th.minimum(trajectories, th.as_tensor(label.shape, device=device))
    trajectories = th.maximum(trajectories, th.zeros(len(label.shape), device=device))

    return trajectories


def advenct_from_quasi_random(
    field: ArrayLike,
    img_shape: Tuple[int, ...],
    n_samples: int,
    invert: bool = True,
    device: Optional[th.device] = None,
) -> ArrayLike:
    """
    Advenct points from quasi random uniform distribution.

    Parameters
    ----------
    field : ArrayLike
        Field array with shape T x D x (Z) x Y x X
    img_shape : Tuple[int, ...]
        Must be D-dimensional.
    n_samples : int
        Number of samples.
    invert : bool
        When true flow is multiplied by -1, resulting in reversal of the flow.
    device : Optional[th.device]
        Torch device, by default uses last GPU if available or mps.

    Returns
    -------
    ArrayLike
        Trajectories of sources N x T x D
    """

    if field.shape[1] != len(img_shape):
        raise ValueError(
            f"Field dimension {field.shape[1]} does not match image shape {len(img_shape)}."
        )

    if device is None:
        device = torch_default_device()

    sources = th.quasirandom.SobolEngine(field.shape[1], seed=42).draw(n_samples)
    sources = sources.to(device)
    sources *= th.as_tensor(img_shape, device=device)

    trajectories = advenct_field(field, sources, img_shape, invert=invert)

    trajectories = th.minimum(trajectories, th.as_tensor(img_shape, device=device))
    trajectories = th.maximum(trajectories, th.zeros(len(img_shape), device=device))

    return trajectories


def trajectories_to_tracks(trajectories: th.Tensor) -> np.ndarray:
    """Converts trajectories to napari tracks format.

    Parameters
    ----------
    trajectories : th.Tensor
        Input N x T x D trajectories.

    Returns
    -------
    np.ndarray
        Napari tracks (N x T) x (2 + D) array.
    """
    trajectories = trajectories.cpu().numpy()
    N, T, D = trajectories.shape

    track_ids = np.repeat(np.arange(N), T)[..., None]
    time_pts = np.tile(np.arange(T), N)[..., None]
    coordinates = trajectories.reshape(-1, D)

    return np.concatenate((track_ids, time_pts, coordinates), axis=1)


def _to_tensor(
    image: ArrayLike,
    channel_axis: Optional[int],
    device: th.device,
) -> th.Tensor:
    """Expands image to 4D tensor with channel axis first and converts to tensor

    Parameters
    ----------
    image : ArrayLike
        Image array.
    channel_axis : Optional[int]
        Channel axis, if None image is expanded to 4D tensor with channel axis first.
    device : th.device
        Torch device.

    Returns
    -------
    th.Tensor
        (C, (Z), Y, X)-tensor.
    """
    image = np.asarray(image)

    if channel_axis is None:
        image = image[np.newaxis, ...]

    elif channel_axis > 0:
        image = np.transpose(
            image,
            (channel_axis, *range(channel_axis), *range(channel_axis + 1, image.ndim)),
        )

    return th.as_tensor(image.astype(np.float32), device=device)


def timelapse_flow(
    images: ArrayLike,
    store_or_path: Union[None, Store, Path, str] = None,
    chunks: Optional[Tuple[int]] = None,
    channel_axis: Optional[int] = None,
    im_factor: int = 4,
    grid_factor: int = 4,
    num_iterations: int = 1000,
    lr: float = 1e-2,
    n_scales: int = 3,
    device: Optional[th.device] = None,
) -> zarr.Array:
    """Compute vector field from timelapse.

    Parameters
    ----------
    images : ArrayLike
        Timelapse images shape as (T, ...).
    store_or_path : Union[None, Store, Path, str], optional
        Zarr storage or output path, if not provided zarr.TempStore is used.
    chunks : Optional[Tuple[int]], optional
        Chunk size, if not provided it chunks time with 1 and the spatial dimensions as big as possible.
    channel_axis : Optional[int], optional
        Channel axis EXCLUDING TIME (first axis), e.g (T, C, Y, X) would have `channel_axis=0`.
        If not provided assumes first axis after time.
    im_factor : int, optional
        Image space down scaling factor, by default 4.
    grid_factor : int, optional
        Grid space down scaling factor, by default 4.
        Grid dimensions will be divided by both `im_factor` and `grid_factor`.
    num_iterations : int, optional
        Number of gradient descent iterations, by default 2000.
    lr : float, optional
        Learning rate (gradient descent step), by default 1e-4
    n_scales : int, optional
        Number of scales used for multi-scale optimization, by default 3.
    device : Optional[th.device], optional
        Torch device, by default uses last GPU if available or mps.

    Returns
    -------
    zarr.Array
        Vector field array with shape (T, D, (Z), Y, X).
    """
    if device is None:
        device = torch_default_device()

    imgs_shape = list(images.shape[1:])
    if channel_axis is not None:
        imgs_shape.pop(channel_axis)

    resized_shape = [max(1, int(s // im_factor)) for s in imgs_shape]
    shape = (images.shape[0], len(imgs_shape), *resized_shape)  # (T, D, (Z), Y, X)
    dtype = np.float16

    if chunks is None:
        chunks = large_chunk_size(shape, dtype=dtype)

    if isinstance(store_or_path, Store):
        output = zarr.zeros(shape, dtype=dtype, store=store_or_path, chunks=chunks)

    else:
        output = create_zarr(
            shape,
            dtype=dtype,
            store_or_path=store_or_path,
            chunks=chunks,
            default_store_type=zarr.TempStore,
        )

    target = _to_tensor(images[0], channel_axis, device)

    for t in tqdm(range(1, len(images)), "Computing flow"):
        source = _to_tensor(images[t], channel_axis, device)

        output[t] = (
            flow_field(
                source,
                target,
                im_factor=im_factor,
                grid_factor=grid_factor,
                lr=lr,
                n_scales=n_scales,
                num_iterations=num_iterations,
            )
            .cpu()
            .numpy()
        )

        target = source

    return output


def add_flow(
    config: MainConfig,
    vector_field: Union[ArrayLike, Sequence[ArrayLike]],
) -> None:
    """
    Adds vector field (coordinate shift) data into nodes.
    If there are fewer vector fields than dimensions, the last dimensions from (z,y,x) have priority.
    For example, if 2 vector fields are provided for a 3D data, only (y, x) are updated.
    Vector field shape, except `t`, can be different from the original image.
    When this happens, the indexing is done mapping the position and rounding.

    Parameters
    ----------
    data_config : DataConfig
        Data configuration parameters.
    vector_field : Sequence[ArrayLike]
        Vector field arrays. Each array per coordinate or a single (T, D, (Z), Y, X)-array.
    """
    LOG.info("Adding shift (vector field) to nodes")

    if isinstance(vector_field, Sequence):
        assert all(v.shape == vector_field[0].shape for v in vector_field)
        ndim = vector_field[0].ndim
        nvecs = len(vector_field)
        vec_shape = np.asarray(vector_field[0].shape)
        is_sequence = True
    else:
        ndim = vector_field.ndim - 1
        nvecs = vector_field.shape[1]
        vec_shape = np.asarray([vector_field.shape[0], *vector_field.shape[2:]])
        is_sequence = False

    shape = np.asarray(config.data_config.metadata["shape"])
    LOG.info(f"Image shape {shape}, ndim={ndim}")
    LOG.info(f"Vec. shape {shape}")
    LOG.info(f"Num. Vec. {nvecs}")

    if len(shape) != ndim:
        raise ValueError(
            "Original data shape and vector field must have same number of dimensions (ignoring channels)."
            f" Found {len(shape)} and {ndim}."
        )

    if np.any(np.abs(vector_field[0]) > 1):
        raise ValueError(
            "Vector field values must be normalized. "
            f"Found {vector_field[0].min()} and {vector_field[0].max()}."
        )

    columns = ["x_shift", "y_shift", "z_shift"]
    coords_scaling = (vec_shape[1:] - 1) / (shape[1:] - 1)
    coordinate_columns = ["z", "y", "x"][-len(coords_scaling) :]
    vec_index_iterator = list(reversed(range(nvecs)))
    # vec_scaling varies depending on the number of vector field and image dimensions
    vec_scaling = np.asarray(shape[1 + len(coords_scaling) - nvecs :])[::-1]

    LOG.info(f"Coordinate_columns {coordinate_columns}")
    LOG.info(f"Vector field scaling of {vec_scaling}")

    engine = sqla.create_engine(config.data_config.database_path)

    for t in tqdm(range(shape[0])):
        with Session(engine) as session:
            query = session.query(NodeDB.id, NodeDB.z, NodeDB.y, NodeDB.x).where(
                NodeDB.t == t
            )
            df = pd.read_sql_query(query.statement, session.bind, index_col="id")

        if len(df) == 0:
            LOG.warning(f"No node found at time point {t}.")
            continue

        coords = df[coordinate_columns].to_numpy()
        coords = np.round(coords * coords_scaling).astype(int)
        coords = np.minimum(
            np.maximum(0, coords), vec_shape[1:] - 1
        )  # truncating boundary
        coords = tuple(coords.T)

        # default value
        df[columns] = 0.0
        # reversed because z could be missing
        for v, colname in zip(vec_index_iterator, columns):
            if is_sequence:
                df[colname] = np.asarray(vector_field[v][t])[
                    coords
                ]  # asarray due lazy loading formats (e.g. dask)
            else:
                df[colname] = np.asarray(vector_field[t, v])[coords]

        # vector field is between -0.5 to 0.5, so it's scaled to the original image size
        # ours is the torch convection divided by 2.0
        # reference
        # https://pytorch.org/docs/stable/generated/torch.nn.functional.grid_sample.html
        df[columns[:nvecs]] *= vec_scaling  # x, y, z

        df["node_id"] = df.index
        with Session(engine) as session:
            statement = (
                sqla.update(NodeDB)
                .where(NodeDB.id == sqla.bindparam("node_id"))
                .values(
                    z_shift=sqla.bindparam("z_shift"),
                    y_shift=sqla.bindparam("y_shift"),
                    x_shift=sqla.bindparam("x_shift"),
                )
            )
            session.connection().execute(
                statement,
                df[["node_id"] + columns].to_dict("records"),
                execution_options={"synchronize_session": False},
            )
            session.commit()
