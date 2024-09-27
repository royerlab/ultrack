from pathlib import Path
from typing import Sequence

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from napari.plugins import _initialize_plugins
from napari.viewer import ViewerModel
from rich import print

from ultrack.cli.utils import (
    layer_key_option,
    napari_reader_option,
    output_directory_option,
    paths_argument,
)
from ultrack.utils.estimation import estimate_parameters_from_labels
from ultrack.utils.printing import pretty_print_df


def _plot_column_over_time(df: pd.DataFrame, column: str, output_dir: Path) -> None:
    """Plots column average over time."""
    df = pd.melt(
        df.groupby("t").agg({column: ["mean", "min", "max"]}),
        var_name="stat",
        value_name=f"stat_{column}",
    )

    sns.set_theme(style="whitegrid")
    plot = sns.lineplot(data=df, palette="tab10")

    fig_path = output_dir / f"{column}_plot.png"
    plot.get_figure().savefig(fig_path)
    plt.close()

    print(f"\n{column} plot saved at {fig_path}")


@click.command("estimate_params")
@paths_argument()
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
    _initialize_plugins()

    viewer = ViewerModel()
    viewer.open(path=paths, plugin=reader_plugin, stack=len(paths) > 1)

    try:
        labels = viewer.layers[int(layer_key)].data
    # conversion to integer might fail
    except (KeyError, ValueError):
        labels = viewer.layers[layer_key].data

    del viewer

    df = estimate_parameters_from_labels(labels, is_timelapse=timelapse)

    covariables = {"area", "distance"}
    covariables = list(covariables.intersection(df.columns))

    output_directory.mkdir(exist_ok=True)

    for col in covariables:
        _plot_column_over_time(df, col, output_directory)

    summary = df[covariables].describe()
    pretty_print_df(summary, title="Parameter summary", row_name="stats")
