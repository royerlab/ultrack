import logging
from typing import Optional, Protocol, Union

import numpy as np
import pandas as pd
import sqlalchemy as sqla
from numpy.typing import ArrayLike
from rich.logging import RichHandler
from sqlalchemy.orm import Session

from ultrack.config.config import MainConfig
from ultrack.core.database import LinkDB, set_node_values
from ultrack.core.linking.features import get_links_features
from ultrack.core.match_gt import UnmatchedNode, match_to_ground_truth
from ultrack.core.segmentation import get_nodes_features

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
LOG.addHandler(RichHandler())


class ProbabilisticClassifier(Protocol):
    def fit(self, X: ArrayLike, y: ArrayLike) -> "ProbabilisticClassifier":
        ...

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        ...


def _validate_classifier(
    classifier: Optional[ProbabilisticClassifier],
) -> ProbabilisticClassifier:
    """
    Validates the classifier if it is not provided.
    """
    if classifier is None:
        try:
            from catboost import CatBoostClassifier
        except ImportError as e:
            raise ImportError(
                "`catboost` is required if classifier is not provided.\n"
                "Please install it with `pip install catboost` or `pip install 'ultrack[ml]'`."
            ) from e
        classifier = CatBoostClassifier(allow_const_label=True)

    return classifier


def add_nodes_prob(
    config: MainConfig,
    indices: ArrayLike,
    probs: ArrayLike,
) -> None:
    """
    Add nodes' probabilities to the segmentation/tracking database.

    Parameters
    ----------
    config : MainConfig
        Main configuration parameters.
    indices : ArrayLike
        Nodes' indices database index.
    probs : ArrayLike
        Nodes' probabilities.
    """
    if probs.ndim == 2:
        probs = probs[:, -1]

    set_node_values(
        config.data_config,
        indices,
        node_prob=probs,
    )


def add_links_prob(
    config: MainConfig,
    indices: ArrayLike,
    probs: ArrayLike,
) -> None:
    """
    Add links' probabilities to the tracking database.

    NOTE: this only updates the weights of the existing links, it does not add new links.
    If you want to add new links, use `add_links` from `ultrack.core.linking.processing`.

    Parameters
    ----------
    config : MainConfig
        Main configuration parameters.
    indices : ArrayLike
        Nx2 array of (target_id, source_id) indices.
    probs : ArrayLike
        Links' probabilities.
    """
    if indices.ndim != 2:
        raise ValueError(f"Indices must be a 2D array, got {indices.shape}")

    if indices.shape[1] != 2:
        raise ValueError(f"Indices must have 2 columns, got {indices.shape[1]}")

    indices = np.asarray(indices, dtype=int)
    engine = sqla.create_engine(config.data_config.database_path)

    with Session(engine) as session:
        stmt = (
            sqla.update(LinkDB)
            .where(
                LinkDB.target_id == sqla.bindparam("target_id"),
                LinkDB.source_id == sqla.bindparam("source_id"),
            )
            .values(
                weight=sqla.bindparam("weight"),
            )
        )
        session.connection().execute(
            stmt,
            [
                {"target_id": t, "source_id": s, "weight": p}
                for t, s, p in zip(*indices, probs)
            ],
            execution_options={"synchronize_session": False},
        )
        session.commit()

    LOG.info("Successfully added %d links probabilities to the database", len(indices))


def predict_nodes_prob(
    config: MainConfig,
    classifier: ProbabilisticClassifier,
    insert_prob: bool = True,
    persistence_features: bool = False,
    coord_features: bool = False,
) -> pd.Series:
    """
    Predicts the probabilities of the nodes' features.

    Parameters
    ----------
    config : MainConfig
        Main configuration parameters.
    classifier : ProbabilisticClassifier
        Probabilistic classifier object.
    insert_prob : bool, optional
        Whether to insert the probabilities to the database, by default True.
    persistence_features : bool, optional
        Whether to include persistence features, by default False.
    coord_features : bool, optional
        Whether to include coordinate (t, (z), y, x) features, by default False.

    Returns
    -------
    pd.Series
        Nodes' probabilities.
    """
    features = get_nodes_features(config, include_persistence=persistence_features)

    if not coord_features:
        features = features.drop(columns=["t", "z", "y", "x"], errors="ignore")

    LOG.info("Predicting classifier with features: %s", str(features.columns))

    probs = classifier.predict_proba(features)

    if probs.ndim == 2:
        probs = probs[:, -1]

    if insert_prob:
        add_nodes_prob(config, features.index, probs)

    return pd.Series(probs, index=features.index)


def fit_nodes_prob(
    config: MainConfig,
    ground_truth: Union[ArrayLike, pd.Series],
    classifier: Optional[ProbabilisticClassifier] = None,
    remove_no_overlap: bool = True,
    insert_prob: bool = True,
    persistence_features: bool = False,
    coord_features: bool = False,
) -> ProbabilisticClassifier:
    """
    Fit a probabilistic classifier to the nodes' features.

    Parameters
    ----------
    config : MainConfig
        Main configuration parameters.
    ground_truth : Union[ArrayLike, pd.Series]
        Ground-truth labels, either a:
        * timelapse array of labels (T, (Z), Y, X) with the same shape as the input data.
        * pandas Series indexed by the nodes' indices.
    classifier : ProbabilisticClassifier
        Probabilistic classifier object.
        Classifier is fit in-place.
        If not provided, it will use `catboost.CatBoostClassifier`.
    remove_no_overlap : bool, optional
        Whether to remove **NO_OVERLAP** nodes (-1) from the ground-truth.
        Classification will compare **matched** (>0) vs **unmatched** nodes (0).
    insert_prob : bool, optional
        Whether to insert the probabilities to the database, by default True.
    persistence_features : bool, optional
        Whether to include persistence features, by default False.
    coord_features : bool, optional
        Whether to include coordinate (t, (z), y, x) features, by default False.

    Returns
    -------
    ProbabilisticClassifier
        Fitted probabilistic classifier.
    """
    features = get_nodes_features(config, include_persistence=persistence_features)

    if not coord_features:
        features = features.drop(columns=["t", "z", "y", "x"], errors="ignore")

    if "id" in features.columns:
        raise ValueError("Features cannot contain 'id' column")

    if not isinstance(ground_truth, (pd.Series, pd.DataFrame)):
        ground_truth = match_to_ground_truth(config, ground_truth)["gt_track_id"]

    # removing cells with no ground-truth, therefore we cannot decide they are blocked or not
    no_overlap = ground_truth != UnmatchedNode.NO_OVERLAP

    LOG.info("Found %f of nodes with no overlap", no_overlap.mean())

    if remove_no_overlap:
        ground_truth = ground_truth[no_overlap]

    # making it a binary classification problem
    ground_truth = (
        ground_truth > UnmatchedNode.BLOCKED
    )  # (includes NO_OVERLAP as negative class)

    LOG.info("Fitting classifier with features: %s", str(features.columns))

    classifier = _validate_classifier(classifier)

    if ground_truth.dtype != bool:
        raise ValueError(
            f"Ground-truth dataframe must be a binary classification problem (bool dtype), got {ground_truth.dtype}"
        )

    # Fit the classifier
    classifier.fit(features.loc[ground_truth.index], ground_truth)

    if insert_prob:
        LOG.info("Adding nodes probabilities to the database")

        probs = classifier.predict_proba(features)
        add_nodes_prob(config, features.index, probs)

    return classifier


def select_competing_links(
    links_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Selects the competing links from the ground-truth.

    Parameters
    ----------
    links_df : pd.DataFrame
        Links dataframe with `gt_link` column.

    Returns
    -------
    pd.DataFrame:
        Links dataframe with the selected competing links.
    """
    links_df = links_df.set_index(["target_id", "source_id"], drop=False)

    has_competing_links = links_df.groupby("target_id")["gt_link"].transform(
        lambda x: x.any()
    )

    links_df = links_df[has_competing_links]

    return links_df


def add_links_gt(
    links_df: pd.DataFrame,
    ground_truth: pd.Series,
) -> pd.Series:
    """
    Adds the ground-truth label to the links dataframe.

    Parameters
    ----------
    links_df : pd.DataFrame
        Links dataframe.
    ground_truth : pd.Series
        Ground-truth labels.

    Returns
    -------
    pd.Series:
        Series with the ground-truth label.
    """
    ground_truth = ground_truth > UnmatchedNode.BLOCKED

    return pd.Series(
        ground_truth.loc[links_df["source_id"]]
        == ground_truth.loc[links_df["target_id"]],
        index=links_df.index,
        name="gt_link",
    )


def fit_links_prob(
    config: MainConfig,
    ground_truth: Union[ArrayLike, pd.Series],
    classifier: Optional[ProbabilisticClassifier] = None,
    remove_no_overlap: bool = True,
    insert_prob: bool = True,
    **kwargs,
) -> ProbabilisticClassifier:
    """
    Fit a probabilistic classifier to the links' features.

    Parameters
    ----------
    config : MainConfig
        Main configuration parameters.
    ground_truth : Union[ArrayLike, pd.Series]
        Ground-truth labels, either a:
        * timelapse array of labels (T, (Z), Y, X) with the same shape as the input data.
        * pandas Series indexed by the nodes' indices.
    classifier : ProbabilisticClassifier
        Probabilistic classifier object.
        Classifier is fit in-place.
        If not provided, it will use `catboost.CatBoostClassifier`.
    remove_no_overlap : bool, optional
        When true it will only consider negative-labeled links when they are competing (overlap)
        with an existing positive-labeled link.
    insert_prob : bool, optional
        Whether to insert the probabilities to the database, by default True.
    **kwargs : dict
        Keyword arguments passed to `get_nodes_features` to compute the features.
        The links' features are the difference between the source and target nodes' features.
    """
    features = get_links_features(config, **kwargs)

    if "id" in features.columns:
        raise ValueError("Features cannot contain 'id' column")

    if not isinstance(ground_truth, (pd.Series, pd.DataFrame)):
        ground_truth = match_to_ground_truth(config, ground_truth)["gt_track_id"]

    links_df = add_links_gt(features, ground_truth)

    if remove_no_overlap:
        links_df = select_competing_links(links_df)

    classifier = _validate_classifier(classifier)

    classifier.fit(links_df.drop(columns=["gt_link"]), links_df["gt_link"])

    if insert_prob:
        LOG.info("Adding links probabilities to the database")

        probs = classifier.predict_proba(links_df.drop(columns=["gt_link"]))
        add_links_prob(
            config,
            links_df[["target_id", "source_id"]].to_numpy(),
            probs,
        )

    return classifier
