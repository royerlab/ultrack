from pathlib import Path

import click
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import sqlalchemy as sqla
from sqlalchemy import func
from sqlalchemy.orm import Session

from ultrack.cli.utils import config_option
from ultrack.config import MainConfig
from ultrack.core.database import LinkDB, NodeDB
from ultrack.core.export.utils import solution_dataframe_from_sql
from ultrack.tracks.graph import add_track_ids_to_tracks_df
from ultrack.utils.constants import NO_PARENT
from ultrack.utils.printing import pretty_print_df


def _nodes_count_over_time(database_path: str, fig_path: Path) -> None:
    """Queries and plots nodes counts over time. Solution isn't taken into account."""
    engine = sqla.create_engine(database_path)
    with Session(engine) as session:
        query = session.query(NodeDB.t, func.count(NodeDB.t)).group_by(NodeDB.t)
        df = pd.read_sql(query.statement, session.bind, index_col="t")

    sns.set_theme(style="whitegrid")
    plot = sns.lineplot(data=df, legend=False)
    plot.set_ylabel("count")

    plot.get_figure().savefig(fig_path)
    plt.close()

    print(f"Nodes count over time saved at {fig_path}")


def q1(arr: pd.Series) -> float:
    return arr.quantile(0.25)


def q2(arr: pd.Series) -> float:
    return arr.median()


def q3(arr: pd.Series) -> float:
    return arr.quantile(0.75)


def _link_stats_over_time(database_path: str, out_dir: Path) -> None:
    """Queries and plots link statistics over time. Solution isn't taken into account."""
    engine = sqla.create_engine(database_path)
    with Session(engine) as session:
        query = session.query(NodeDB.t, LinkDB.weight).join(
            NodeDB, NodeDB.id == LinkDB.source_id
        )
        groups = pd.read_sql(query.statement, session.bind).groupby("t")

    sns.set_theme(style="whitegrid")

    fig_path = out_dir / "link_weight_plot.png"
    df = pd.melt(
        groups.agg({"weight": [q1, q2, q3]}),
        var_name="quantile",
        value_name="link_weight",
    )
    plot = sns.lineplot(data=df, legend=False)
    plot.set_ylabel("link weight")
    plot.get_figure().savefig(fig_path)
    plt.close()

    print(f"Links weights over time saved at {fig_path}")

    fig_path = out_dir / "link_count_plot.png"
    plot = sns.lineplot(data=groups.count(), legend=False)
    plot.set_ylabel("count")
    plot.get_figure().savefig(fig_path)
    plt.close()

    print(f"Links count over time saved at {fig_path}")


def _solution_summary(database_path: str) -> None:
    """Computes some statistics from the solution."""
    df = solution_dataframe_from_sql(database_path)
    df = add_track_ids_to_tracks_df(df)

    total = len(df)
    no_parents = (df["parent_id"] == NO_PARENT).sum()
    divisions = (df.groupby("parent_id").size() > 1).sum() - int(
        no_parents > 0
    )  # subtracting NO_PARENTS
    ends = total - df.index.isin(df["parent_id"]).sum()
    tracks = len(df["track_id"].unique())

    print("\n")
    pretty_print_df(
        pd.DataFrame(
            [
                [total],
                [no_parents],
                [ends],
                [divisions],
                [tracks],
            ],
            columns=["count"],
            index=["segments", "appearance", "disappearance", "division", "tracks"],
        ),
        title="Solution summary",
    )


@click.command("data_summary")
@config_option()
@click.option(
    "--output-directory",
    "-o",
    required=True,
    type=click.Path(path_type=Path),
    default="summary",
    show_default=True,
    help="Output plots directory.",
)
def data_summary_cli(
    config: MainConfig,
    output_directory: Path,
) -> None:
    """Prints a summary of the database data."""

    database_path = config.data_config.database_path
    output_directory.mkdir(exist_ok=True)

    _nodes_count_over_time(database_path, output_directory / "nodes_count.png")
    _link_stats_over_time(database_path, output_directory)
    _solution_summary(database_path)
