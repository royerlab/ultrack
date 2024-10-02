from typing import Callable

import numpy as np
import pandas as pd
from napari.viewer import ViewerModel

from ultrack.utils.constants import NO_PARENT
from ultrack.validation.link_validation import Annotation, LinkValidation


def test_link_validation(
    make_napari_viewer: Callable[[], ViewerModel],
) -> None:

    tracks_df = pd.DataFrame(
        {
            "id": [0, 1, 2, 3, 4, 5],
            "parent_id": [NO_PARENT, 0, 1, 2, 1, 4],
            "track_id": [1, 1, 2, 2, 3, 3],
            "parent_track_id": [NO_PARENT, NO_PARENT, 1, 1, 1, 1],
            "t": [0, 1, 2, 3, 2, 3],
            "z": [0, 0, 0, 0, 0, 0],
            "y": [10, 20, 30, 10, 20, 10],
            "x": [10, 20, 30, 10, 20, 10],
        }
    )

    image = np.random.randint(255, size=(5, 64, 64))

    viewer = make_napari_viewer()
    widget = LinkValidation(
        image,
        tracks_df=tracks_df,
        viewer=viewer,
    )
    widget.btn_done.clicked.connect(viewer.close)

    assert not widget.btn_undo.isEnabled()
    widget.btn_correct.click()
    assert widget.btn_undo.isEnabled()

    widget.btn_skip.click()

    widget.btn_undo.click()
    widget.btn_undo.click()

    assert np.all(widget.annotations["annotation"] == Annotation.UNLABELED)
    assert not widget.btn_undo.isEnabled()

    widget.btn_incorrect.click()
    assert widget.btn_undo.isEnabled()

    # reseting
    widget.btn_undo.click()

    # add annotations
    widget.annotations = pd.DataFrame(
        {"id": [1, 2], "annotation": [Annotation.CORRECT, Annotation.INCORRECT]}
    )
    assert widget.current_idx == 2

    widget.btn_correct.click()
