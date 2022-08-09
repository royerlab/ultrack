import click

from ultrack.cli.compute import compute_cli
from ultrack.cli.export import export_cli
from ultrack.cli.initialize import initialize_cli


@click.group()
def main():
    pass


main.add_command(initialize_cli)
main.add_command(compute_cli)
main.add_command(export_cli)
