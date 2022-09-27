from pathlib import Path
from typing import Sequence

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from napari.viewer import ViewerModel
from rich import print
from rich.table import Table

from ultrack.cli.utils import (
    layer_key_option,
    napari_reader_option,
    output_directory_option,
)
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


def _plot_column_over_time(df: pd.DataFrame, column: str, output_dir: Path) -> None:
    """Plots column average over time."""
    df = df.groupby("t").agg({column: ["mean", "min", "max"]})

    sns.set_theme(style="whitegrid")
    plot = sns.lineplot(data=df, palette="tab10")

    fig_path = output_dir / f"{column}_plot.png"
    plot.get_figure().savefig(fig_path)
    plt.close()

    print(f"\n{column} plot saved at {fig_path}")


@click.command("estimate_params")
@click.argument("paths", nargs=-1, type=click.Path(path_type=Path))
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
@output_directory_option(
    default=".",
    show_default=True,
    help="Plots output directory",
)
def estimate_params_cli(
    paths: Sequence[Path],
    reader_plugin: str,
    layer_key: str,
    timelapse: bool,
    output_directory: Path,
) -> None:
    """Helper command to estimate a few parameters from labeled data."""

    viewer = ViewerModel()
    viewer.open(path=paths, plugin=reader_plugin, stack=len(paths) > 1)

    try:
        labels = viewer.layers[int(layer_key)].data
    # conversion to integer might fail
    except (KeyError, ValueError):
        labels = viewer.layers[layer_key].data

    df = estimate_parameters_from_labels(labels, is_timelapse=timelapse)

    covariables = {"area", "distance"}
    covariables = list(covariables.intersection(df.columns))

    output_directory.mkdir(exist_ok=True)

    for col in covariables:
        _plot_column_over_time(df, col, output_directory)

    summary = df[covariables].describe()
    _print_df(summary)
