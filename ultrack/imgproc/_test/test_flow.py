import pytest
import torch as th

from ultrack.imgproc.flow import advenct_field, flow, to_tracks


@pytest.mark.parametrize("ndim", [2, 3])
def test_flow_field(ndim: int, request) -> None:
    intensity = 1_000
    size = (64,) * ndim
    sigma = 15
    im_factor = 2
    grid_factor = 4

    grid = th.stack(th.meshgrid([th.arange(s) for s in size], indexing="ij"), dim=-1)

    mus = th.Tensor(
        [[0.5, 0.5, 0.5], [0.55, 0.5, 0.5], [0.57, 0.48, 0.53], [0.55, 0.45, 0.55]]
    )[:, :ndim]

    mus = (mus * th.tensor(size)).round().int()

    frames = th.stack(
        [intensity * th.exp(-th.square(grid - mu).sum(dim=-1) / sigma) for mu in mus]
    )

    fields = flow(
        frames.numpy(),
        im_factor=im_factor,
        grid_factor=grid_factor,
        num_iterations=1000,
    )
    trajectory = advenct_field(fields, mus[None, 0], size)
    tracks = to_tracks(trajectory)

    if request.config.getoption("--show-napari-viewer"):
        import napari

        kwargs = {"blending": "additive", "interpolation3d": "nearest", "rgb": False}

        viewer_1 = napari.Viewer()
        # viewer.add_image(fields.cpu().numpy(), colormap="turbo", scale=(2, 2, 2))
        viewer_1.add_image(frames.cpu().numpy(), **kwargs)
        viewer_1.add_tracks(tracks)

        viewer_2 = napari.Viewer()
        viewer_2.add_image(
            th.clamp_min(fields, 0).cpu().numpy(),
            colormap="red",
            scale=(im_factor,) * 3,
            **kwargs,
        )
        viewer_2.add_image(
            th.clamp_min(-fields, 0).cpu().numpy(),
            colormap="blue",
            scale=(im_factor,) * 3,
            **kwargs,
        )

        napari.run()

    assert th.allclose(trajectory, mus.half(), atol=0.5, rtol=0.0)
