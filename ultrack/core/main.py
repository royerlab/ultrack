from typing import Literal, Optional, Sequence, Union

from numpy.typing import ArrayLike

from ultrack.config import MainConfig
from ultrack.core.database import clear_all_data
from ultrack.core.linking.processing import link
from ultrack.core.linking.utils import clear_linking_data
from ultrack.core.segmentation.processing import segment
from ultrack.core.solve.processing import solve
from ultrack.core.solve.sqltracking import SQLTracking
from ultrack.imgproc.flow import add_flow
from ultrack.utils.edge import labels_to_edges


def track(
    config: MainConfig,
    *,
    labels: Optional[ArrayLike] = None,
    sigma: Optional[Union[Sequence[float], float]] = None,
    detection: Optional[ArrayLike] = None,
    edges: Optional[ArrayLike] = None,
    images: Sequence[ArrayLike] = tuple(),
    scale: Optional[Sequence[float]] = None,
    vector_field: Optional[Union[ArrayLike, Sequence[ArrayLike]]] = None,
    overwrite: Literal["all", "links", "solutions", "none", True, False] = "none",
) -> None:
    """
    All-in-one function for cell tracking, it accepts multiple inputs (labels or edges)
    and run all intermediate steps, computing segmentation hypothesis, linking and solving the ILP.
    The results must be queried using the export function of preference.

    Note: Either `labels` or `detection` and `edges` can be used as input, but not both.

    Parameters
    ----------
    config : MainConfig
        Tracking configuration parameters.
    labels : Optional[ArrayLike], optional
        Segmentation labels of shape (T, (Z), Y, X), by default None
    sigma : Optional[Union[Sequence[float], float]], optional
        Edge smoothing parameter (gaussian blur) for labels to edges conversion, by default None
    detection : Optional[ArrayLike], optional
        Fuzzy detection array of shape (T, (Z), Y, X), by default None
    edges : Optional[ArrayLike], optional
        Edges array of shape (T, (Z), Y, X), by default None
    images : Sequence[ArrayLike]
        Optinal sequence of images (T, (Z), Y, X) for color space filtering.
    scale : Sequence[float]
        Optional scaling for nodes' distances.
    vector_field : Union[ArrayLike, Sequence[ArrayLike]]
        Vector field arrays. Each array per coordinate or a single (T, D, (Z), Y, X)-array.
    overwrite : Literal["all", "links", "solutions", "none"], optional
        Clear the corresponding data from the database, by default nothing is overwritten with "none"
        When not "none", only the cleared and subsequent parts of the pipeline is executed.
    """
    if labels is not None and (detection is not None or edges is not None):
        raise ValueError(
            "`labels` and `detection` or `edges` cannot be supplied at the same time."
        )

    if labels is not None:
        detection, edges = labels_to_edges(labels, sigma=sigma)
    elif detection is None or edges is None:
        raise ValueError(
            "Both `detection` and `edges` must be supplied when not using `labels`."
        )

    if isinstance(overwrite, bool):
        overwrite = "all" if overwrite else "none"

    overwrite = overwrite.lower()

    if overwrite == "all":
        clear_all_data(config.data_config.database_path)

    elif overwrite == "links":
        clear_linking_data(config.data_config.database_path)

    elif overwrite == "solutions":
        SQLTracking.clear_solution_from_database(config.data_config.database_path)

    elif overwrite != "none":
        raise ValueError(
            f"Overwrite option {overwrite} not found. Expected one of 'all', 'links', 'solutions', 'none'."
        )

    if overwrite in ("all", "none"):
        segment(
            detection,
            edges,
            config,
        )

    if overwrite in ("all", "links", "none"):
        link(config, images=images, scale=scale)
        if vector_field is not None:
            add_flow(config, vector_field)

    solve(config)
