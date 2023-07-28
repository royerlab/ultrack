from typing import Optional, Sequence, Union

from numpy.typing import ArrayLike

from ultrack.config import MainConfig
from ultrack.core.linking.processing import link
from ultrack.core.segmentation.processing import segment
from ultrack.core.solve.processing import solve
from ultrack.utils.edge import labels_to_edges


def track(
    config: MainConfig,
    *,
    labels: Optional[ArrayLike] = None,
    sigma: Optional[Union[Sequence[float], float]] = None,
    detection: Optional[ArrayLike] = None,
    edges: Optional[ArrayLike] = None,
    scale: Optional[Sequence[float]] = None,
    overwrite: bool = False,
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
    scale : Sequence[float]
        Optional scaling for nodes' distances.
    overwrite : bool, optional
        Cleans up segmentation, linking and tracking content before processing, by default False
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

    segment(
        detection,
        edges,
        config,
        overwrite=overwrite,
    )

    link(config, scale=scale)
    solve(config)
