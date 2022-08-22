from typing import Optional, Sequence, Union

from ultrack.config.dataconfig import DataConfig
from ultrack.config.trackingconfig import TrackingConfig
from ultrack.core.tracking.sqltracking import SQLTracking


def track(
    tracking_config: TrackingConfig,
    data_config: DataConfig,
    indices: Optional[Union[Sequence[int], int]] = None,
) -> None:
    """Compute tracking by selecting nodes with maximum flow from database.

    Parameters
    ----------
    tracking_config : TrackingConfig
        Tracking configuration parameters.
    data_config : DataConfig
        Data configuration parameters.
    indices : Optional[Sequence[int], int], optional
        Batch indices for tracking a subset of nodes, by default everything is tracked.
    """
    tracker = SQLTracking(tracking_config, data_config)

    if indices is None:
        indices = range(0, tracker.num_batches)

    elif isinstance(indices, int):
        indices = [indices]

    for index in indices:
        tracker(index=index)
