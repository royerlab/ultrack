import logging
from typing import Optional

import napari
import numpy as np
from magicgui.widgets import Container, PushButton, create_widget
from napari.layers import Tracks

from ultrack.tracks.sorting import sort_trees_by_length
from ultrack.widgets.utils import wait_cursor

LOG = logging.getLogger(__name__)


class TrackInspectionWidget(Container):
    def __init__(self, viewer: napari.Viewer) -> None:
        super().__init__()

        self._viewer = viewer
        self._current_track_layer = None

        self._sorted_tracks = []
        self._tree_index = 0

        self._tracks_layer_w = create_widget(annotation=Tracks)
        self._tracks_layer_w.changed.connect(self._on_layer_change)

        self._next_btn = PushButton(text="Next", enabled=False)
        self._next_btn.changed.connect(self._on_next)

        self._prev_btn = PushButton(text="Prev", enabled=False)
        self._prev_btn.changed.connect(self._on_prev)

        self.append(self._tracks_layer_w)
        self.append(self._next_btn)
        self.append(self._prev_btn)

    @property
    def tree_index(self) -> int:
        return self._tree_index

    @tree_index.setter
    def tree_index(self, value: int) -> int:
        if 0 <= value < len(self._sorted_tracks):
            self._tree_index = value
            self._prev_btn.enabled = value > 0
            self._next_btn.enabled = value < len(self._sorted_tracks) - 1
            if self._current_track_layer is not None:
                subtree = self._sorted_tracks[self._tree_index].to_numpy()
                self._current_track_layer.data = subtree
                self._current_track_layer.name = f"subtree {int(subtree[0, 0])}"

    def _on_next(self) -> None:
        self.tree_index += 1

    def _on_prev(self) -> None:
        self.tree_index -= 1

    def _on_layer_change(self, layer: Optional[Tracks]) -> None:
        LOG.info(f"Track inspection layer update: {layer}")
        if self._current_track_layer is not None:
            self._viewer.layers.remove(self._current_track_layer.name)

        if layer is None:
            return

        with wait_cursor():
            self._sorted_tracks = sort_trees_by_length(layer.data, layer.graph)

        self._current_track_layer = self._viewer.add_tracks(
            np.zeros((2, 4), dtype=float),
            scale=layer.scale[-3:],
            translate=layer.translate[-3:],
            colormap="twilight",
        )
        self.tree_index = 0
