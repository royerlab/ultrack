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
    def fit(self, X: ArrayLike, y: ArrayLike) -> "ProbabilisticClassifier": ...

    def predict_proba(self, X: ArrayLike) -> ArrayLike: ...


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
        classifier = CatBoostClassifier(allow_const_label=True, silent=True)

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

    if probs.ndim == 2:
        probs = probs[:, -1]

    probs_df = pd.DataFrame(
        probs,
        index=pd.MultiIndex.from_arrays(
            [indices[:, 0], indices[:, 1]],
            names=["target_id", "source_id"],
        ),
        columns=["weight"],
    )

    LOG.info(
        "Adding %d links probabilities to the database",
        len(indices),
    )

    engine = sqla.create_engine(config.data_config.database_path)

    with Session(engine) as session:
        query = session.query(LinkDB.id, LinkDB.target_id, LinkDB.source_id)
        link_df = pd.read_sql(
            query.statement, session.bind, index_col=["target_id", "source_id"]
        )

        update = (
            sqla.update(LinkDB)
            .where(LinkDB.id == sqla.bindparam("link_id"))
            .values(weight=sqla.bindparam("weight"))
        )
        probs_df = probs_df.join(link_df, how="left")

        assert not probs_df.isna().any().any()

        session.connection().execute(
            update,
            [
                {"link_id": link_id, "weight": weight}
                for link_id, weight in zip(probs_df["id"], probs_df["weight"])
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
        LOG.info(
            "Predicting nodes probabilities with features: %s", str(features.columns)
        )

        probs = classifier.predict_proba(features)
        add_nodes_prob(config, features.index, probs)

    return classifier


def select_competing_links(
    links_df: pd.DataFrame,
    gt_link: pd.Series,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Selects the competing links from the ground-truth.

    Parameters
    ----------
    links_df : pd.DataFrame
        Links dataframe with features.
    gt_link : pd.Series
        Ground-truth (0/1)-links series.

    Returns
    -------
    tuple[pd.DataFrame, pd.Series]
        Links dataframe and ground-truth with the selected competing links.
    """
    links_df["gt_link"] = gt_link
    target_ids = links_df.index.get_level_values(0)

    has_competing_links = links_df.groupby(target_ids)["gt_link"].transform(
        lambda x: x.any()
    )
    LOG.info(
        "Found %d competing links out of %d candidates",
        has_competing_links.sum(),
        len(target_ids),
    )

    links_df = links_df[has_competing_links]

    return links_df.drop(columns=["gt_link"]), links_df["gt_link"]


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
    target_ids = links_df.index.get_level_values(0)
    source_ids = links_df.index.get_level_values(1)

    matched_nodes = ground_truth > UnmatchedNode.BLOCKED
    matched_nodes = (
        matched_nodes.loc[source_ids].to_numpy()
        & matched_nodes.loc[target_ids].to_numpy()
    )

    correct_links = (
        ground_truth.loc[source_ids].to_numpy()
        == ground_truth.loc[target_ids].to_numpy()
    )

    gt_links = pd.Series(
        correct_links & matched_nodes,
        index=pd.MultiIndex.from_arrays(
            [target_ids, source_ids], names=["target_id", "source_id"]
        ),
        name="gt_link",
    )

    LOG.info(
        "Found %d ground-truth links out of %d candidates",
        gt_links.sum(),
        len(gt_links),
    )

    return gt_links


def predict_links_prob(
    config: MainConfig,
    classifier: ProbabilisticClassifier,
    *,
    insert_prob: bool = True,
    persistence_features: bool = False,
    **kwargs,
) -> pd.Series:
    """
    Predicts the probabilities of the links' features.

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
    **kwargs : dict
        Keyword arguments passed to `get_nodes_features` to compute the features.
        The links' features are the difference between the source and target nodes' features.
    """
    features = get_links_features(
        config,
        include_persistence=persistence_features,
        **kwargs,
    )

    for col in ["target_id", "source_id"]:
        if col in features.columns:
            raise ValueError(f"Features cannot contain '{col}' column")

    LOG.info("Predicting links probabilities with features: %s", str(features.columns))

    probs = classifier.predict_proba(features)

    if probs.ndim == 2:
        probs = probs[:, -1]

    if insert_prob:
        add_links_prob(
            config,
            np.asarray(features.index.to_list()),
            probs,
        )

    return pd.Series(probs, index=features.index, name="link_prob")


def fit_links_prob(
    config: MainConfig,
    ground_truth: Union[ArrayLike, pd.Series],
    classifier: Optional[ProbabilisticClassifier] = None,
    remove_no_overlap: bool = True,
    insert_prob: bool = True,
    persistence_features: bool = False,
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
    persistence_features : bool, optional
        Whether to include persistence features, by default False.
    **kwargs : dict
        Keyword arguments passed to `get_nodes_features` to compute the features.
        The links' features are the difference between the source and target nodes' features.
    """
    features = get_links_features(
        config,
        include_persistence=persistence_features,
        **kwargs,
    )

    for col in ["target_id", "source_id"]:
        if col in features.columns:
            raise ValueError(f"Features cannot contain '{col}' column")

    if not isinstance(ground_truth, (pd.Series, pd.DataFrame)):
        ground_truth = match_to_ground_truth(config, ground_truth)["gt_track_id"]

    LOG.info("Ground-truth labels %s: %s", ground_truth.shape, ground_truth)
    LOG.info("Features %s: %s", features.shape, features.columns)

    gt_link = add_links_gt(features, ground_truth)

    if remove_no_overlap:
        features, gt_link = select_competing_links(features, gt_link)
        if features.empty:
            raise ValueError("Dataset is empty after removing no overlap links")

    classifier = _validate_classifier(classifier)

    classifier.fit(features, gt_link)

    if insert_prob:
        LOG.info(
            "Predicting links probabilities with features: %s", str(features.columns)
        )

        probs = classifier.predict_proba(features)

        LOG.info("Predicted probabilities %s: %s", probs.shape, probs)

        add_links_prob(
            config,
            np.asarray(features.index.to_list()),
            probs,
        )

    return classifier
