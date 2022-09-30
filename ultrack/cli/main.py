import os

if os.environ.get("ULTRACK_DEBUG", False):
    import logging

    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

import click

from ultrack.cli.clear_database import clear_database_cli
from ultrack.cli.config import config_cli
from ultrack.cli.data_summary import data_summary_cli
from ultrack.cli.estimate_params import estimate_params_cli
from ultrack.cli.export import export_cli
from ultrack.cli.labels_to_edges import labels_to_edges_cli
from ultrack.cli.link import link_cli
from ultrack.cli.segment import segmentation_cli
from ultrack.cli.track import track_cli


@click.group()
def main():
    pass


main.add_command(clear_database_cli)
main.add_command(config_cli)
main.add_command(data_summary_cli)
main.add_command(estimate_params_cli)
main.add_command(export_cli)
main.add_command(labels_to_edges_cli)
main.add_command(link_cli)
main.add_command(segmentation_cli)
main.add_command(track_cli)
