import logging
import math as m
from pathlib import Path
from typing import Optional, Tuple, Union, cast

import numpy as np
import torch as th
import torch.nn.functional as F
import zarr
from numpy.typing import ArrayLike
from skimage.measure import regionprops_table
from tqdm import tqdm
from zarr.storage import Store

from ultrack.imgproc.utils import create_zarr
from ultrack.utils import large_chunk_size
from ultrack.utils.constants import ULTRACK_DEBUG
from ultrack.utils.cuda import import_module, torch_default_device

LOG = logging.getLogger(__name__)

try:
    import cupy as xp

    LOG.info("cupy found.")

except (ModuleNotFoundError, ImportError):
    import numpy as xp

    LOG.info("cupy not found using numpy and scipy.")


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
            array = xp.asarray(tensor.squeeze())
            blurred = ndi.gaussian_filter(
                array,
                sigma=0.5 / scale_factor,
                output=array,
            )
            tensor = th.as_tensor(blurred, device=tensor.device)
            tensor = tensor.reshape(orig_shape)
            LOG.info(f"Antialiasing with sigma = {0.5 / scale_factor}.")

    return F.interpolate(tensor, **kwargs, mode=mode, align_corners=True)


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
    return F.affine_grid(T, grid_shape, align_corners=True)


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
    lr: float = 1e-4,
    n_scales: int = 3,
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
        Learning rate (gradient descent step), by default 1e-4
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

    source = source.unsqueeze(0)
    target = target.unsqueeze(0)

    device = source.device
    scales = np.flip(np.power(2, np.arange(n_scales)))
    grid = None

    for scale in scales:
        with th.no_grad():
            scaled_im_factor = im_factor * scale
            scaled_source = _interpolate(
                source, scale_factor=1 / scaled_im_factor, antialias=True
            )
            scaled_target = _interpolate(
                target, scale_factor=1 / scaled_im_factor, antialias=True
            )

            if grid is None:
                grid_shape = tuple(
                    m.ceil(s / grid_factor) for s in scaled_source.shape[-ndim:]
                )
                grid = identity_grid(grid_shape).to(device)
                grid0 = grid.clone()
            else:
                grid = _interpolate_grid(grid, scale_factor=2)
                grid0 = identity_grid(grid.shape[-(ndim + 1) : -1]).to(device)

        grid.requires_grad_(True).retain_grad()

        LOG.info(f"scale: {scale}")
        LOG.info(f"image shape: {scaled_source.shape}")
        LOG.info(f"grid shape: {grid.shape}")

        for i in range(num_iterations):
            if grid.grad is not None:
                grid.grad.zero_()

            large_grid = _interpolate_grid(grid, size=scaled_source.shape[-ndim:])

            im2hat = F.grid_sample(target, large_grid, align_corners=True)
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

    transformed_image = F.grid_sample(image[None], field, align_corners=True)

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
        int_sources = th.minimum(int_sources, field_shape - 1).int()
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
    num_iterations: int = 2000,
    lr: float = 1e-4,
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

    if chunks is None:
        chunks = large_chunk_size(shape, dtype=np.float16)

    if isinstance(store_or_path, Store):
        output = zarr.zeros(shape, dtype=np.float16, store=store_or_path, chunks=chunks)

    else:
        output = create_zarr(
            shape,
            dtype=np.float16,
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
