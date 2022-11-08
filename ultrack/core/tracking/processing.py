import logging
from typing import Optional

from ultrack.config.dataconfig import DataConfig
from ultrack.config.trackingconfig import TrackingConfig
from ultrack.core.tracking.sqltracking import SQLTracking

LOG = logging.getLogger(__name__)


def track(
    tracking_config: TrackingConfig,
    data_config: DataConfig,
    batch_index: Optional[int] = None,
    overwrite: bool = False,
) -> None:
    """Compute tracking by selecting nodes with maximum flow from database.

    Parameters
    ----------
    tracking_config : TrackingConfig
        Tracking configuration parameters.
    data_config : DataConfig
        Data configuration parameters.
    batch_index : Optional[int], optional
        Batch index for processing a subset of nodes, by default everything is processed.
    overwrite : bool, optional
        Resets existing solution before processing.
    """
    tracker = SQLTracking(tracking_config, data_config)

    if overwrite and (batch_index is None or batch_index == 0):
        tracker.reset_solution()

    if isinstance(batch_index, int):
        LOG.info(f"Tracking batch index {batch_index}")
        tracker(index=batch_index)
    else:
        LOG.info(f"Tracking batch indices from 0 to {tracker.num_batches}")
        for batch_index in range(tracker.num_batches):
            tracker(index=batch_index)
