import logging
from enum import Enum
from typing import Optional, Tuple

import napari
import numpy as np
import pandas as pd
from numpy.typing import ArrayLike
from qtpy.QtCore import Qt
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import QLabel, QPushButton, QVBoxLayout, QWidget

from ultrack.utils.constants import NO_PARENT

LOG = logging.getLogger(__name__)


class Annotation(Enum):
    """
    Enum class representing possible annotation states.

    Attributes
    ----------
    UNLABELED : int
        Represents an unlabeled state.
    CORRECT : int
        Represents a correct annotation.
    INCORRECT : int
        Represents an incorrect annotation.
    SKIPPED : int
        Represents a skipped annotation.
    """

    UNLABELED = 0
    CORRECT = 1
    INCORRECT = 2
    SKIPPED = 3


class LinkValidation(QWidget):
    """
    A QWidget subclass for validating and annotating tracks.

    Parameters
    ----------
    images : ArrayLike
        The image data as an array.
    tracks_df : pd.DataFrame
        Dataframe containing track data.
        Must contain columns: "id", "parent_track_id", "parent_id", "t", "y", "x".
    viewer : napari.Viewer
        The napari viewer instance.
    crop_size : int, optional
        Size for cropping the image, by default 64.
    scale : Optional[ArrayLike], optional
        Scale of the image, by default None which means ones are used.
    parent : Optional[QWidget], optional
        Parent widget, by default None.
    """

    def __init__(
        self,
        *images: ArrayLike,
        tracks_df: pd.DataFrame,
        viewer: napari.Viewer,
        crop_size: int = 64,
        scale: Optional[ArrayLike] = None,
        parent: Optional[QWidget] = None,
    ):
        super().__init__(parent)

        self._current_idx = 0
        self._rng = np.random.default_rng(1)

        self._annotation_df = pd.DataFrame(
            {"annotation": Annotation.UNLABELED}, index=_filter_non_pairs(tracks_df)
        )

        size = self._annotation_df.shape[0]
        self._ordering = self._rng.integers(size, size=(size,))

        self._tracks_df = tracks_df.set_index("id", drop=False)
        if "z" not in self._tracks_df.columns:
            self._tracks_df["z"] = 0

        self._images = list(images)

        if scale is None:
            self._scale = np.ones(self.ndim)
        else:
            self._scale = np.asarray(scale)

        self.setLayout(self._createLayout())

        # setups viewer
        self._viewer = viewer
        self._crop_size = np.broadcast_to(crop_size, shape=self.ndim)
        self._crop_size = self._crop_size * self._scale / self._scale.max()
        self._crop_size = self._crop_size.astype(int)
        self._half_crop = self._crop_size // 2

        # setups layers
        empty_image = np.zeros(self._crop_size, dtype=np.uint8)
        kwargs = dict(blending="additive", scale=self._scale)

        self._frame_layers = [
            self._viewer.add_image(empty_image, name="frame", colormap="red", **kwargs)
            for _ in range(len(self._images))
        ]
        self._prev_frame_layers = [
            self._viewer.add_image(
                empty_image, name="prev. frame", colormap="green", **kwargs
            )
            for _ in range(len(self._images))
        ]

        for i in range(len(self._images)):
            self._frame_layers[i]._keep_auto_contrast = True
            self._prev_frame_layers[i]._keep_auto_contrast = True

        self._pts_layer = self._viewer.add_points(
            scale=scale,
            ndim=self.ndim,
            name="centrois",
            out_of_slice_display=True,
            face_color="transparent",
        )

        # setups view
        self.current_idx = 0

    def _createLayout(self) -> QVBoxLayout:
        """
        Creates and returns a QVBoxLayout containing UI elements for the widget.

        Returns
        -------
        QVBoxLayout
            The created layout containing buttons and labels for the widget.
        """
        layout = QVBoxLayout()

        # Create buttons
        self.btn_correct = QPushButton("Correct", self)
        self.btn_incorrect = QPushButton("Incorrect", self)
        self.btn_skip = QPushButton("Skip", self)
        self.btn_undo = QPushButton("Undo", self)
        self.btn_done = QPushButton("Done", self)
        self.lb_counter = QLabel(self.counter_text, self)

        # Set tools tips
        self.btn_correct.setToolTip("Mark pair as correct (Shortcut: 1)")
        self.btn_incorrect.setToolTip("Mark pair as incorrect (Shortcut: 2)")
        self.btn_skip.setToolTip("Skip pair (Shortcut: 3)")
        self.btn_undo.setToolTip("Undo last annotation (Shortcut: 4)")
        self.btn_done.setToolTip("Close viewer and save annotations")

        # Set shortcuts
        self.btn_correct.setShortcut(QKeySequence(Qt.Key_1))
        self.btn_incorrect.setShortcut(QKeySequence(Qt.Key_2))
        self.btn_skip.setShortcut(QKeySequence(Qt.Key_3))
        self.btn_undo.setShortcut(QKeySequence(Qt.Key_4))

        # Add buttons to the layout
        layout.addWidget(self.btn_correct)
        layout.addWidget(self.btn_incorrect)
        layout.addWidget(self.btn_skip)
        layout.addWidget(self.btn_undo)
        layout.addWidget(self.btn_done)
        layout.addWidget(self.lb_counter)

        # Add actions
        self.btn_correct.clicked.connect(
            lambda: self._update_annotation(Annotation.CORRECT, 1)
        )
        self.btn_incorrect.clicked.connect(
            lambda: self._update_annotation(Annotation.INCORRECT, 1)
        )
        self.btn_undo.clicked.connect(
            lambda: self._update_annotation(Annotation.UNLABELED, -1)
        )
        self.btn_skip.clicked.connect(
            lambda: self._update_annotation(Annotation.SKIPPED, 1)
        )

        return layout

    @property
    def counter_text(self) -> str:
        """
        Returns the text displaying the current counter status.
        """
        i = self.current_idx + 1
        return f"Sample: {i} ({i / self._annotation_df.shape[0]:0.3f} %)"

    @property
    def ndim(self) -> int:
        return self._images[0].ndim - 1

    @property
    def current_idx(self) -> int:
        return self._current_idx

    @current_idx.setter
    def current_idx(self, value: int) -> None:
        """
        Setter for the current index. Also updates button states and view.

        Parameters
        ----------
        value : int
            New current index to set from 0 to number of samples - 1.
        """
        self.btn_correct.setEnabled(True)
        self.btn_incorrect.setEnabled(True)
        self.btn_skip.setEnabled(True)
        self.btn_undo.setEnabled(True)

        if value == 0:
            self.btn_undo.setEnabled(False)

        elif value == self._annotation_df.shape[0] - 1:
            self.btn_correct.setEnabled(False)
            self.btn_incorrect.setEnabled(False)
            self.btn_skip.setEnabled(False)

        self._current_idx = value
        LOG.info(f"Current index: {self._current_idx}")
        self.lb_counter.setText(self.counter_text)
        self._update_view()

    def _update_annotation(
        self,
        annotation: Annotation,
        increment: int,
    ) -> None:
        """
        Updates the annotation for the current index and logs the annotation.

        Parameters
        ----------
        annotation : Annotation
            The new annotation value.
        increment : int
            Amount to increment the current index by after updating the annotation.
        """
        self._annotation_df.iloc[self._ordering[self._current_idx]] = annotation
        LOG.info(f"Annotation: {annotation} to {self._current_idx}")
        if increment != 0:
            self.current_idx += increment

    def _crop_image(
        self, image: ArrayLike, t: int, coords: ArrayLike
    ) -> Tuple[ArrayLike, ArrayLike]:
        """
        Crop image around provided coordinates.

        Parameters
        ----------
        image : ArrayLike
            Image to crop.
        t : int
            Time point.
        coords : ArrayLike
            Coordinates around which to crop.

        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            Cropped image and lower bounds of the cropped region.
        """
        coords = np.asarray(coords, dtype=int)[-self.ndim :]

        # Compute crop bounds
        shape = np.asarray(image.shape[-self.ndim :])
        lower = np.clip(coords - self._half_crop, 0, None)
        upper = np.clip(coords + self._crop_size - self._half_crop, None, shape - 1)

        # Crop image
        slicing = (t,) + tuple(slice(l, u) for l, u in zip(lower, upper))
        crop = np.asarray(image[slicing])

        return crop, lower

    def _update_view(self) -> None:
        """
        Updates the viewer layers based on the current index and related track data.
        """
        node = self._tracks_df.iloc[self._ordering[self._current_idx]]
        parent_node = self._tracks_df.loc[int(node["parent_id"])]

        # Update frame layer
        for l1, l2, im in zip(
            self._frame_layers, self._prev_frame_layers, self._images
        ):
            l1.data, offset = self._crop_image(
                im, node["t"].item(), node[["z", "y", "x"]]
            )
            l2.data, _ = self._crop_image(
                im, parent_node["t"].item(), node[["z", "y", "x"]]
            )

        # Update points layer
        centorids = (
            np.stack([node[["z", "y", "x"]], parent_node[["z", "y", "x"]]])[
                :, -self.ndim :
            ]
            - offset[None, :]
        )
        self._pts_layer.data = centorids
        self._pts_layer.edge_color = np.array([[1, 0, 0], [0, 1, 0]])

        # Center view
        positions = self._scale * centorids[0, -len(self._scale) :]
        n_dims = len(self._viewer.dims.order)
        start_dim = n_dims - self.ndim

        self._viewer.dims.set_point(range(start_dim, n_dims), positions)

    @property
    def annotations(self) -> pd.DataFrame:
        return self._annotation_df

    @annotations.setter
    def annotations(self, value: pd.DataFrame) -> None:
        """
        Update annotations by mergin with existing annotations and moving to the next unlabeled sample.

        Parameters
        ----------
        value : pd.DataFrame
            Dataframe with annotations.
        """
        if "id" in value.columns:
            value = value.set_index("id", drop=False)["annotation"]

        self._annotation_df.update(value)
        labels = self._annotation_df["annotation"].to_numpy()
        annotated = labels != Annotation.UNLABELED
        annot_ids = np.where(annotated)[0]
        not_annot_ids = np.where(~annotated)[0]
        self._rng.shuffle(not_annot_ids)
        self._ordering = np.concatenate([annot_ids, not_annot_ids])
        self.current_idx = len(annot_ids)


def _filter_non_pairs(tracks_df: pd.DataFrame) -> ArrayLike:
    """Filter out divisions, appearing or disappearing tracks."""

    tracks_df = tracks_df.set_index("id", drop=False)

    # Filter out appearing cells
    tracks_df = tracks_df[tracks_df["parent_id"] != NO_PARENT]

    # Filter division events
    parents_track_ids = tracks_df["parent_track_id"]
    tracks_df = tracks_df[tracks_df["parent_track_id"] == parents_track_ids]

    return tracks_df["id"].to_numpy(dtype=int)


def validate_links(
    *images: ArrayLike,
    tracks_df: pd.DataFrame,
    annotations: Optional[pd.DataFrame] = None,
    crop_size: int = 64,
    scale: Optional[ArrayLike] = None,
    viewer: Optional[napari.Viewer] = None,
) -> pd.DataFrame:
    """
    Validates links by creating a GUI interface in napari for manual annotation.

    Parameters
    ----------
    images : ArrayLike
        Images data as an array.
    tracks_df : pd.DataFrame
        Dataframe containing track data.
    annotations : Optional[pd.DataFrame], optional
        Dataframe containing previous annotations, by default None.
    crop_size : int, optional
        Size for cropping the image, by default 64.
    scale : Optional[ArrayLike], optional
        Scale of the image, by default None which means ones are used.
    viewer : Optional[napari.Viewer], optional
        Existing napari viewer instance, by default None, in which case a new viewer is created.

    Returns
    -------
    pd.DataFrame
        Dataframe containing annotations from the validation session.
    """

    if viewer is None:
        viewer = napari.Viewer()

    widget = LinkValidation(
        *images,
        tracks_df=tracks_df,
        viewer=viewer,
        crop_size=crop_size,
        scale=scale,
    )

    widget.btn_done.clicked.connect(viewer.close)

    if annotations is not None:
        widget.annotations = annotations

    viewer.window.add_dock_widget(widget, area="right")

    napari.run()
