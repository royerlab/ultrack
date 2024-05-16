import click

from ultrack.cli.check_gurobi import check_gurobi_cli
from ultrack.cli.clear_database import clear_database_cli
from ultrack.cli.config import config_cli
from ultrack.cli.data_summary import data_summary_cli
from ultrack.cli.estimate_params import estimate_params_cli
from ultrack.cli.export import export_cli
from ultrack.cli.flow import add_flow_cli
from ultrack.cli.labels_to_edges import labels_to_contours_cli
from ultrack.cli.link import link_cli
from ultrack.cli.segment import segmentation_cli
from ultrack.cli.server import server_cli
from ultrack.cli.solve import solve_cli


@click.group()
def main():
    pass


main.add_command(add_flow_cli)
main.add_command(check_gurobi_cli)
main.add_command(clear_database_cli)
main.add_command(config_cli)
main.add_command(data_summary_cli)
main.add_command(estimate_params_cli)
main.add_command(export_cli)
main.add_command(labels_to_contours_cli)
main.add_command(link_cli)
main.add_command(segmentation_cli)
main.add_command(solve_cli)
main.add_command(server_cli)
