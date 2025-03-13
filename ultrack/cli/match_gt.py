import logging
from pathlib import Path
from typing import Optional, Sequence

import click
import cloudpickle
import toml
from napari.plugins import _initialize_plugins
from napari.viewer import ViewerModel
from rich.logging import RichHandler

from ultrack.cli.segment import _get_layer_data
from ultrack.cli.utils import (
    batch_index_option,
    config_option,
    napari_reader_option,
    overwrite_option,
    paths_argument,
    persistence_option,
)
from ultrack.config import MainConfig
from ultrack.core.match_gt import match_to_ground_truth
from ultrack.ml.classification import fit_links_prob, fit_nodes_prob

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
LOG.addHandler(RichHandler())


@click.command("match_gt")
@paths_argument()
@napari_reader_option()
@config_option()
@click.option(
    "--ground-truth-layer",
    "-gl",
    required=False,
    type=str,
    default=None,
    help="Ground-truth layer index on napari.",
)
@click.option(
    "--output-model",
    "-om",
    type=click.Path(dir_okay=False, path_type=Path),
    required=False,
    default=None,
    help="Optional output model file path.",
)
@click.option(
    "--output-config",
    "-oc",
    type=click.Path(dir_okay=False, path_type=Path),
    help="Optional output config file path.",
)
@click.option(
    "--is-segmentation",
    is_flag=True,
    type=bool,
    default=False,
    help="Indicates ground-truth are fully curated segmentation masks. "
    "When activated different costs are used for insertions and deletions.",
)
@click.option(
    "--is-tracking",
    is_flag=True,
    type=bool,
    default=False,
    help="Indicates ground-truth are tracking instances results.",
)
@click.option(
    "--is-dense",
    is_flag=True,
    type=bool,
    default=False,
    help="Indicates ground-truth are dense annotations (everything is annotated).",
)
@click.option(
    "--insert-prob",
    is_flag=True,
    type=bool,
    default=False,
    help="Insert estimated probabilities into the database.",
)
@batch_index_option()
@overwrite_option()
@persistence_option()
def match_gt_cli(
    paths: Sequence[Path],
    reader_plugin: str,
    config: MainConfig,
    ground_truth_layer: Optional[str],
    output_model: Optional[Path],
    output_config: Optional[Path],
    is_segmentation: bool,
    is_tracking: bool,
    is_dense: bool,
    insert_prob: bool,
    batch_index: Optional[int],
    overwrite: bool,
    persistence: bool,
) -> None:
    """
    Match ground-truth labels to the segmentation/tracking database.
    """

    if output_model is not None and output_model.exists() and not overwrite:
        raise FileExistsError(
            f"Output model {output_model} already exists. Use --overwrite to overwrite."
        )

    if output_config is not None:
        if not is_segmentation:
            raise ValueError(
                "Output config is only available for segmentation ground-truth `--is-segmentation`."
            )

        if output_config.exists() and not overwrite:
            raise FileExistsError(
                f"Output config {output_config} already exists. Use --overwrite to overwrite."
            )

    # Data loading
    _initialize_plugins()

    viewer = ViewerModel()
    viewer.open(path=paths, plugin=reader_plugin)

    if ground_truth_layer is None:
        if len(viewer.layers) > 1:
            raise ValueError(
                "Multiple layers found, please specify `--ground-truth-layer`."
            )
        else:
            ground_truth_layer = viewer.layers[0].name

    gt = _get_layer_data(viewer, ground_truth_layer)

    # Match ground-truth to database
    gt_df, new_config = match_to_ground_truth(
        config=config,
        gt_labels=gt,
        scale=config.data_config.metadata.get("scale"),
        is_segmentation=is_segmentation,
        optimize_config=True,
        batch_index=batch_index,
    )

    if output_config is not None:
        LOG.info("Estimated new config: %s", new_config)
        LOG.info("Saving new config to %s", output_config)
        with open(output_config, "w") as f:
            toml.dump(new_config.model_dump(by_alias=True), f)

    if insert_prob or output_model is not None:
        model = fit_nodes_prob(
            config,
            gt_df["gt_track_id"],
            persistence_features=persistence,
            insert_prob=insert_prob,
            remove_no_overlap=not is_dense,
        )

        if is_tracking:
            link_model = fit_links_prob(
                config,
                gt_df["gt_track_id"],
                persistence_features=persistence,
                insert_prob=insert_prob,
            )
            model = {"nodes": model, "links": link_model}

        if output_model is not None:
            LOG.info("Saving model to %s", output_model)
            with open(output_model, "wb") as f:
                cloudpickle.dump(model, f)
