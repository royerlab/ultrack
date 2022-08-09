import click

from ultrack.cli.compute import compute
from ultrack.cli.export import export
from ultrack.cli.initialize import initialize


@click.group()
def main():
    pass


main.add_command(initialize)
main.add_command(compute)
main.add_command(export)
