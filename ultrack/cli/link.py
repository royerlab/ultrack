from pathlib import Path
from typing import Optional

import click
from napari.viewer import ViewerModel

from ultrack import link
from ultrack.cli.utils import config_option, overwrite_option
from ultrack.config import MainConfig


@click.command("link")
@config_option()
@overwrite_option()
@click.option(
    "--image-path", "-i", default=None, type=click.Path(path_type=Path), help="TODO"
)
def link_cli(config: MainConfig, overwrite: bool, image_path: Optional[Path]) -> None:
    """Links segmentation candidates adjacent in time."""

    # FIXME: temporary solution for testing
    image = None
    if image_path is not None:
        viewer = ViewerModel()
        image = viewer.open(image_path)[0].data[:2]

    link(
        config.linking_config,
        config.data_config,
        overwrite=overwrite,
        image=image,
        channel_axis=0,
    )
