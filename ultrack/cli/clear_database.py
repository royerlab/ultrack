import click

from ultrack.cli.utils import config_option
from ultrack.config.config import MainConfig
from ultrack.core.database import clear_all_data
from ultrack.core.linking.utils import clear_linking_data
from ultrack.core.solve.sqltracking import SQLTracking


@click.command("clear_database")
@click.argument("mode", type=click.Choice(["all", "links", "solutions"]))
@config_option()
def clear_database_cli(mode: str, config: MainConfig) -> None:
    """Cleans database content."""

    database_path = config.data_config.database_path
    if mode == "all":
        clear_all_data(database_path)
    elif mode == "links":
        clear_linking_data(database_path)
    elif mode == "solutions":
        SQLTracking.clear_solution_from_database(database_path)
    else:
        raise NotImplementedError(f"Clear database mode {mode} not implemented.")
