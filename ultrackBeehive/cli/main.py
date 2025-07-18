import click

from ultrackBeehive.cli.check_gurobi import check_gurobi_cli
from ultrackBeehive.cli.clear_database import clear_database_cli
from ultrackBeehive.cli.config import config_cli
from ultrackBeehive.cli.data_summary import data_summary_cli
from ultrackBeehive.cli.estimate_params import estimate_params_cli
from ultrackBeehive.cli.export import export_cli
from ultrackBeehive.cli.flow import add_flow_cli
from ultrackBeehive.cli.labels_to_edges import labels_to_contours_cli
from ultrackBeehive.cli.link import link_cli
from ultrackBeehive.cli.match_gt import match_gt_cli
from ultrackBeehive.cli.predict import add_probs_cli
from ultrackBeehive.cli.segment import segmentation_cli
from ultrackBeehive.cli.server import server_cli
from ultrackBeehive.cli.solve import solve_cli
from ultrackBeehive.cli.view import view_cli


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
main.add_command(match_gt_cli)
main.add_command(add_probs_cli)
main.add_command(segmentation_cli)
main.add_command(solve_cli)
main.add_command(server_cli)
main.add_command(view_cli)
