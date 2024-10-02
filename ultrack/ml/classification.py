from numpy.typing import ArrayLike

from ultrack.config.config import MainConfig
from ultrack.core.database import set_node_values


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
    set_node_values(
        config.data_config,
        indices,
        node_prob=probs,
    )
