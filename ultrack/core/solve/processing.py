import logging
from typing import Optional

from ultrack.config.config import MainConfig
from ultrack.core.solve.sqltracking import SQLTracking

LOG = logging.getLogger(__name__)


def solve(
    config: MainConfig,
    batch_index: Optional[int] = None,
    overwrite: bool = False,
    use_annotations: bool = False,
) -> None:
    """Compute tracking by selecting nodes with maximum flow from database.

    Parameters
    ----------
    config : MainConfig
        Configuration parameters.
    batch_index : Optional[int], optional
        Batch index for processing a subset of nodes, by default everything is processed.
    overwrite : bool, optional
        Resets existing solution before processing.
    use_annotations : bool, optional
        Use annotations to fix ILP variables, by default False
    """
    tracker = SQLTracking(config)

    if overwrite and (batch_index is None or batch_index == 0):
        tracker.reset_solution()

    if use_annotations:
        if not overwrite:
            LOG.warning(
                "Annotations are being used without overwriting the existing solution.\n"
                "Ignore this warning if you this is your first tracking call."
            )

    if isinstance(batch_index, int):
        LOG.info(f"Solving ILP of batch index {batch_index}")
        tracker(index=batch_index, use_annotations=use_annotations)
    else:
        LOG.info(f"Solving ILP batch indices from 0 to {tracker.num_batches}")
        # interleaved processing
        for batch_index in range(0, tracker.num_batches, 2):
            tracker(index=batch_index, use_annotations=use_annotations)
        for batch_index in range(1, tracker.num_batches, 2):
            tracker(index=batch_index, use_annotations=use_annotations)
