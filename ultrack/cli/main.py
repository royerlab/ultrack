import click

from ultrack.cli.export import export_cli
from ultrack.cli.link import link_cli
from ultrack.cli.segment import segmentation_cli
from ultrack.cli.track import track_cli


@click.group()
def main():
    pass


main.add_command(segmentation_cli)
main.add_command(link_cli)
main.add_command(track_cli)
main.add_command(export_cli)
