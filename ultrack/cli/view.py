import click
import napari

from ultrack.cli.utils import config_option
from ultrack.config import MainConfig
from ultrack.widgets.hierarchy_viz_widget import HierarchyVizWidget


@click.group("view")
def view_cli() -> None:
    """View data in napari."""


@view_cli.command("hierarchy")
@config_option()
def view_hierarchy_cli(config: MainConfig) -> None:
    """Opens napari viewer with hierarchy visualization widget."""
    viewer = napari.Viewer()
    widget = HierarchyVizWidget(viewer, config)
    viewer.window.add_dock_widget(widget)
    napari.run()
