from pathlib import Path

import click
import pandas as pd
from napari.viewer import ViewerModel
from rich import print
from rich.table import Table

from ultrack.cli.utils import layer_key_option, napari_reader_option
from ultrack.utils.estimation import estimate_parameters_from_labels


def _print_df(df: pd.DataFrame) -> None:
    """Converts dataframe to rich table and prints it."""
    table = Table(title="Parameters summary")

    table.add_column("stats")
    for c in df.columns:
        table.add_column(c)

    for name, row in df.iterrows():
        row = [str(value) for value in row]
        table.add_row(name, *row)

    print(table)


@click.command("estimate_params")
@click.argument("path", nargs=1, type=click.Path(path_type=Path))
@napari_reader_option()
@layer_key_option()
@click.option(
    "--timelapse",
    "-t/-nt",
    default=True,
    type=bool,
    is_flag=True,
    show_default=True,
    help="Indicates if data is a timelapse.",
)
def estimate_params_cli(
    path: Path,
    reader_plugin: str,
    layer_key: str,
    timelapse: bool = True,
) -> None:
    """Helper command to estimate a few parameters from labeled data."""

    viewer = ViewerModel()
    viewer.open(path=path, plugin=reader_plugin)

    try:
        labels = viewer.layers[int(layer_key)].data
    # conversion to integer might fail
    except (KeyError, ValueError):
        labels = viewer.layers[layer_key].data

    df = estimate_parameters_from_labels(labels, is_timelapse=timelapse)

    summary = df.describe()
    _print_df(summary)
