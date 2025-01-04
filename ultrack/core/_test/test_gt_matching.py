from typing import Tuple

import zarr

from ultrack.config import MainConfig
from ultrack.core.gt_matching import match_to_ground_truth


def test_match_to_ground_truth(
    segmentation_database_mock_data: MainConfig,
    timelapse_mock_data: Tuple[zarr.Array, zarr.Array, zarr.Array],
) -> None:

    config = segmentation_database_mock_data

    _, _, gt = timelapse_mock_data

    df_gt, opt_cfg = match_to_ground_truth(
        config,
        gt,
        track_id_graph={},
        segmentation_gt=True,
        optimize_config=True,
    )

    for c in ["gt_track_id", "gt_parent_track_id"]:
        assert c in df_gt.columns

    assert config.linking_config.max_distance != opt_cfg.linking_config.max_distance
    assert config.segmentation_config.min_area != opt_cfg.segmentation_config.min_area
    assert config.segmentation_config.max_area != opt_cfg.segmentation_config.max_area
    assert (
        config.segmentation_config.min_frontier
        != opt_cfg.segmentation_config.min_frontier
    )
