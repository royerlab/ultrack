import numpy as np
import pandas as pd

from ultrack.utils.tracks import sort_trees_by_length


def test_sortrees_by_length() -> None:
    """
                 5
                 --
            2   /
         -----
     1  /      \\
    ---          -----
       \\         6
         --
         3

        --------
        4
    """
    df = pd.DataFrame(
        {
            "track_id": [
                1,
                1,
                1,
                2,
                2,
                2,
                2,
                2,
                3,
                3,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                4,
                5,
                5,
                6,
                6,
                6,
                6,
                6,
            ],
        }
    )
    graph = {6: 2, 5: 2, 2: 1, 3: 1}

    sorted = sort_trees_by_length(df, graph)

    assert len(sorted) == 2

    for track_id in [1, 2, 3, 5, 6]:
        assert track_id in sorted[0]["track_id"]
        assert track_id not in sorted[1]["track_id"]

    assert not np.any(4 == sorted[0]["track_id"])
    assert np.all(4 == sorted[1]["track_id"])

    reconstr_df = pd.concat(sorted).sort_values("track_id")
    assert np.allclose(reconstr_df, df)
