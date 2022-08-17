import click

from ultrack.cli.compute import compute_cli
from ultrack.cli.export import export_cli
from ultrack.cli.segment import segmentation_cli


@click.group()
def main():
    pass


main.add_command(segmentation_cli)
main.add_command(compute_cli)
main.add_command(export_cli)
