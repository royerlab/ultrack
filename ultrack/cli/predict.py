from pathlib import Path
from typing import Literal

import click
from cloudpickle import load

from ultrack.cli.utils import config_option, persistense_option
from ultrack.config import MainConfig
from ultrack.ml.classification import predict_nodes_prob


@click.command("add_probs")
@click.argument("classif_pickle_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--var",
    type=click.Choice(["nodes", "edges", "divisions", "appearances", "disappearances"]),
    default="nodes",
    help="Variable to assign probabilities.",
)
@config_option()
@persistense_option()
def add_probs_cli(
    classif_pickle_path: Path,
    var: Literal["nodes", "edges", "divisions", "appearances", "disappearances"],
    config: MainConfig,
    persistense: bool,
) -> None:
    """Predicts and adds nodes' probabilities to the database."""

    with open(classif_pickle_path, "rb") as f:
        classifier = load(f)

    if var == "nodes":
        predict_nodes_prob(config, classifier, persistense_features=persistense)
    else:
        # TODO add edges and other probabilities
        raise NotImplementedError(f"Variable {var} not implemented.")
