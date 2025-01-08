import logging
from typing import Optional, Protocol, Union

import pandas as pd
from numpy.typing import ArrayLike
from rich.logging import RichHandler

from ultrack.config.config import MainConfig
from ultrack.core.database import set_node_values
from ultrack.core.match_gt import match_to_ground_truth
from ultrack.core.segmentation import get_nodes_features

LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
LOG.addHandler(RichHandler())


class ProbabilisticClassifier(Protocol):
    def fit(self, X: ArrayLike, y: ArrayLike) -> "ProbabilisticClassifier":
        ...

    def predict_proba(self, X: ArrayLike) -> ArrayLike:
        ...


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
    classifier : ProbabilisticClassifier
        Probabilistic classifier object.
        Classifier is fit in-place.
        If not provided, it will use `xgboost.XGBClassifier`.
    ground_truth : Union[ArrayLike, pd.Series]
        Ground-truth labels, either a:
        * timelapse array of labels (T, (Z), Y, X) with the same shape as the input data.
        * pandas Series indexed by the nodes' indices.
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
        ground_truth = match_to_ground_truth(config, ground_truth)
        ground_truth = ground_truth["gt_track_id"] > 0

    LOG.info("Fitting classifier with features: %s", str(features.columns))

    if classifier is None:
        try:
            from xgboost import XGBClassifier
        except ImportError as e:
            raise ImportError(
                "`xgboost` is required if classifier is not provided.\n"
                "Please install it with `pip install xgboost` or `pip install 'ultrack[ml]'`."
            ) from e
        classifier = XGBClassifier(objective="binary:logistic")

    if ground_truth.dtype != bool:
        raise ValueError(
            f"Ground-truth dataframe must be a binary classification problem (bool dtype), got {ground_truth.dtype}"
        )

    training_labels = ground_truth.loc[features.index]

    if training_labels.all():
        LOG.warning(
            "All labels are True, inverting the labels.\n"
            "ARE YOU SURE THIS IS CORRECT?\n"
            "Check if ground-truth matching is correct.\n"
            "Considering it an anomaly detection problem."
        )
        training_labels = ~training_labels

    # Fit the classifier
    classifier.fit(features, training_labels)

    if insert_prob:
        LOG.info("Adding probabilities to the database")

        probs = classifier.predict_proba(features)
        add_nodes_prob(config, features.index, probs)

    return classifier
