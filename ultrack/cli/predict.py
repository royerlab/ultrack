from pathlib import Path
from typing import Literal

import click
from cloudpickle import load

from ultrack.cli.utils import config_option, persistence_option
from ultrack.config import MainConfig
from ultrack.ml.classification import predict_links_prob, predict_nodes_prob


@click.command("add_probs")
@click.argument("classif_pickle_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--var",
    type=click.Choice(["nodes", "links", "divisions", "appearances", "disappearances"]),
    default="nodes",
    help="Variable to assign probabilities.",
)
@config_option()
@persistence_option()
def add_probs_cli(
    classif_pickle_path: Path,
    var: Literal["nodes", "links", "divisions", "appearances", "disappearances"],
    config: MainConfig,
    persistence: bool,
) -> None:
    """Predicts and adds nodes' probabilities to the database."""

    with open(classif_pickle_path, "rb") as f:
        classifier = load(f)
        if isinstance(classifier, dict):
            classifier = classifier[var]

    if var == "nodes":
        predict_nodes_prob(config, classifier, persistence_features=persistence)
    elif var == "links":
        predict_links_prob(config, classifier, persistence_features=persistence)
    else:
        # TODO add edges and other probabilities
        raise NotImplementedError(f"Variable {var} not implemented.")
