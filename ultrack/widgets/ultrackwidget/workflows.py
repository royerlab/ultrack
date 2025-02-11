import logging
from enum import Enum
from typing import Dict, Generator, List, Optional, Sequence

import napari
import numpy as np
import pandas as pd
import zarr
from napari.layers import Image, Labels, Layer, Tracks
from napari.types import ArrayLike

from ultrack import MainConfig, link, segment, solve, to_tracks_layer, tracks_to_zarr
from ultrack.imgproc import detect_foreground, robust_invert
from ultrack.imgproc.flow import (
    add_flow,
    advenct_from_quasi_random,
    timelapse_flow,
    trajectories_to_tracks,
)
from ultrack.tracks import tracks_df_movement
from ultrack.utils import labels_to_contours
from ultrack.utils.array import array_apply, create_zarr
from ultrack.utils.cuda import on_gpu
from ultrack.widgets.ultrackwidget.utils import UltrackInput

LOGGER = logging.getLogger(__name__)


class WorkflowChoice(Enum):
    AUTO_DETECT = "Auto detect foreground and contours from image"
    AUTO_FROM_LABELS = "Auto detect contours from labels"
    MANUAL = "Manual selection of inputs"


class WorkflowStage(Enum):
    """
    Enumeration representing different stages of the workflow.
    """

    PREPROCESSING = "Preprocessing"
    SEGMENTATION = "Segmentation"
    LINKING = "Linking"
    SOLVING = "Tracking"
    DONE = "Done"

    def next(self):
        """
        Get the next stage in the workflow.

        Returns
        -------
        WorkflowStage
            The next stage.
        """
        members = list(WorkflowStage)
        idx = members.index(self)
        idx = min(idx + 1, len(members) - 1)
        return members[idx]


class UltrackWorkflow:
    """
    A class representing the Ultrack workflow, managing the various stages
    from preprocessing to tracking.

    Attributes
    ----------
    viewer : napari.Viewer
        The napari viewer instance.
    config : Optional[MainConfig]
        The main configuration for the workflow.
    """

    def __init__(self, viewer: napari.Viewer) -> None:
        """
        Initialize the workflow with the given viewer.

        Parameters
        ----------
        viewer : napari.Viewer
            The napari viewer instance.
        """
        self.config: Optional[MainConfig] = None
        self.additional_options = {}
        self.viewer = viewer
        self.last_reached_stage = None
        self.inputs_used = {}

    def get_stage(
        self,
        config: MainConfig,
        additional_options: dict,
        inputs: dict[UltrackInput, Layer],
    ) -> WorkflowStage:
        """
        Determine the current stage of the workflow based on the configuration.

        Parameters
        ----------
        config : MainConfig
            The main configuration for the workflow.
        additional_options : Dict
            Additional options for preprocessing.
        inputs : dict[UltrackInput, Layer]
            The layers used in the workflow.

        Returns
        -------
        WorkflowStage
            The current stage of the workflow.
        """
        if (
            self.config is None
            or self.config != config
            or self.additional_options != additional_options
            or self.last_reached_stage == WorkflowStage.PREPROCESSING
            or self.inputs_used != inputs
        ):
            return WorkflowStage.PREPROCESSING
        elif (
            self.config.segmentation_config != config.segmentation_config
            or self.last_reached_stage == WorkflowStage.SEGMENTATION
        ):
            return WorkflowStage.SEGMENTATION
        elif (
            self.config.linking_config != config.linking_config
            or self.last_reached_stage == WorkflowStage.LINKING
        ):
            return WorkflowStage.LINKING
        elif (
            self.config.tracking_config != config.tracking_config
            or self.last_reached_stage == WorkflowStage.SOLVING
        ):
            return WorkflowStage.SOLVING
        else:
            return WorkflowStage.DONE

    def notify_input_modification(self) -> None:
        """
        Notify that the input has been modified, resetting the configuration.
        """
        self.config = None
        self.additional_options = {}
        self.last_reached_stage = None
        self.inputs_used = {}

    def run(
        self,
        config: MainConfig,
        workflow_choice: WorkflowChoice,
        inputs: Dict[UltrackInput, Layer],
        additional_options: Dict,
    ) -> Generator:
        """
        Run the workflow based on the provided configuration and workflow choice.

        Parameters
        ----------
        config : MainConfig
            The main configuration for the workflow.
        workflow_choice : str
            The workflow choice indicating the preprocessing method.
        additional_options : Dict
            Additional options for preprocessing.
        inputs: Dict[UltrackInput, Layer]
            The inputs for the workflow.
        """
        image = inputs.get(UltrackInput.IMAGE, None)
        labels = inputs.get(UltrackInput.LABELS, None)
        detection = inputs.get(UltrackInput.DETECTION, None)
        contours = inputs.get(UltrackInput.CONTOURS, None)

        stage = self.get_stage(config, additional_options, inputs)
        self.inputs_used = inputs
        self.config = config
        self.additional_options = additional_options
        self.last_reached_stage = WorkflowStage.PREPROCESSING
        try:
            if stage == WorkflowStage.PREPROCESSING:
                if workflow_choice != WorkflowChoice.MANUAL:
                    if image is None and labels is None:
                        raise ValueError(
                            "Image or labels must be provided in auto mode"
                        )

                    if image is not None:
                        scale = image.scale
                    else:
                        scale = labels.scale

                    detection, contours = self._run_preprocessing(
                        image=image.data if image is not None else None,
                        labels=labels.data if labels is not None else None,
                        choice=workflow_choice,
                        **additional_options,
                    )

                    detection = Image(
                        data=detection, name="Foreground", visible=False, scale=scale
                    )
                    contours = Image(
                        data=contours, name="Contours", visible=False, scale=scale
                    )
                    yield detection
                    yield contours
                else:
                    if detection is None or contours is None:
                        raise ValueError(
                            "Foreground and contours must be provided in manual mode"
                        )
                stage = stage.next()
        except Exception as e:
            self.last_reached_stage = WorkflowStage.PREPROCESSING
            raise e

        try:
            if stage == WorkflowStage.SEGMENTATION:
                self._run_segmentation(detection, contours)
                tracks_layer, flow_field = self._add_flow(
                    image, workflow_choice, additional_options
                )
                if tracks_layer is not None:
                    yield tracks_layer
                if flow_field is not None:
                    add_flow(self.config, flow_field)

                stage = stage.next()

        except Exception as e:
            self.last_reached_stage = WorkflowStage.SEGMENTATION
            raise e

        try:
            if stage == WorkflowStage.LINKING:
                self._run_linking(detection.scale)
                stage = stage.next()
        except Exception as e:
            self.last_reached_stage = WorkflowStage.LINKING
            raise e

        try:
            if stage == WorkflowStage.SOLVING:
                self._run_solving()
                tracks_df, graph = to_tracks_layer(self.config)

                segments = tracks_to_zarr(
                    config,
                    tracks_df,
                    store_or_path=zarr.TempStore(suffix="segments"),
                    overwrite=True,
                )

                yield Labels(
                    data=segments,
                    name="Segments",
                    visible=True,
                    scale=detection.scale,
                )

                is_3d = segments.ndim == 4
                dims = ["z", "y", "x"] if is_3d else ["y", "x"]

                yield Tracks(
                    data=tracks_df[["track_id", "t", *dims]].values,
                    graph=graph,
                    name="Tracks",
                    scale=detection.scale[1:],
                )
        except Exception as e:
            self.last_reached_stage = WorkflowStage.SOLVING
            raise e
        self.last_reached_stage = WorkflowStage.DONE
        return

    def _run_preprocessing(
        self,
        image: Optional[ArrayLike],
        labels: Optional[ArrayLike],
        choice: WorkflowChoice,
        detect_foreground_kwargs: Optional[Dict] = None,
        robust_invert_kwargs: Optional[Dict] = None,
        label_to_contours_kwargs: Optional[Dict] = None,
        **_,
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Run the preprocessing step based on the chosen method.

        Parameters
        ----------
        image : Optional[ArrayLike]
            The input image for preprocessing.
        labels : Optional[ArrayLike]
            The input labels for preprocessing.
        choice : str
            The preprocessing choice.
        detect_foreground_kwargs : Dict
            Keyword arguments for detecting foreground.
        robust_invert_kwargs : Dict
            Keyword arguments for robust inversion.
        label_to_contours_kwargs : Dict
            Keyword arguments for converting labels to contours.

        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            The detection and contours arrays.
        """
        if choice == WorkflowChoice.AUTO_DETECT:
            if detect_foreground_kwargs is None:
                LOGGER.info("No detect_foreground_kwargs provided, using default.")
                detect_foreground_kwargs = {}
            if robust_invert_kwargs is None:
                LOGGER.info("No robust_invert_kwargs provided, using default.")
                robust_invert_kwargs = {}
            return self._auto_detect(
                image, detect_foreground_kwargs, robust_invert_kwargs
            )
        elif choice == WorkflowChoice.AUTO_FROM_LABELS:
            if label_to_contours_kwargs is None:
                LOGGER.info("No label_to_contours_kwargs provided, using default.")
                label_to_contours_kwargs = {}
            return self._auto_from_labels(labels, label_to_contours_kwargs)
        else:
            raise ValueError(f"Unknown choice {choice}")

    def _add_flow(
        self, image: Layer, workflow_choice: WorkflowChoice, additional_options: dict
    ) -> tuple[Optional[Tracks], Optional[ArrayLike]]:
        """Add flow vectors to ultrack if provided.

        Parameters
        ----------
        image : ArrayLike
            The input image.
        workflow_choice : str
            The workflow choice. Either "manual" or "auto_detect".
        additional_options : dict
            Additional options for the workflow.

        Returns
        -------
        Optional[Tracks]
            The flow vectors as tracks.
        Optional[ArrayLike]
            The flow field.
        """
        if (
            workflow_choice == WorkflowChoice.MANUAL
            or workflow_choice == WorkflowChoice.AUTO_DETECT
        ):
            if additional_options.get("flow_kwargs") is not None:
                flow_kwargs = additional_options["flow_kwargs"]
                if flow_kwargs["__enable__"]:
                    del flow_kwargs["__enable__"]
                    flow_field = timelapse_flow(
                        image.data,
                        store_or_path=zarr.TempStore(suffix="flow"),
                        **flow_kwargs,
                    )
                    flow_kwargs["__enable__"] = True

                    shape = list(image.data.shape)
                    if flow_kwargs["channel_axis"]:
                        shape.pop(flow_kwargs["channel_axis"] + 1)

                    ndim = flow_field.shape[1]
                    dd = ["dz", "dy", "dx"] if ndim == 3 else ["dy", "dx"]
                    d = ["z", "y", "x"] if ndim == 3 else ["y", "x"]
                    trajectory = advenct_from_quasi_random(
                        flow_field, shape[-ndim:], n_samples=1000
                    )

                    flow_tracklets = pd.DataFrame(
                        trajectories_to_tracks(trajectory),
                        columns=["track_id", "t", *d],
                    )
                    flow_tracklets[d] += 0.5
                    #                       ^^^ napari was crashing otherwise,
                    #                           might be an openGL issue
                    flow_tracklets[dd] = tracks_df_movement(
                        flow_tracklets, cols=tuple(d)
                    )
                    flow_tracklets["angles"] = np.arctan2(
                        # [dy, dx] or [dz, dy, dx]
                        *[flow_tracklets[d] for d in dd]
                    )

                    tracks = Tracks(
                        flow_tracklets[["track_id", "t", *d]],
                        name="flow vectors",
                        visible=False,
                        tail_length=25,
                        features=flow_tracklets[["angles", "dy", "dx"]],
                        colormap="hsv",
                    )
                    tracks.color_by = "angles"
                    return tracks, flow_field
        return None, None

    def _auto_detect(
        self,
        image: ArrayLike,
        detect_foreground_kwargs: Dict,
        robust_invert_kwargs: Dict,
    ) -> tuple[ArrayLike, ArrayLike]:
        """
        Automatically detect foreground and contours from the input image.

        Parameters
        ----------
        image : ArrayLike
            The input image.
        detect_foreground_kwargs : Dict
            Keyword arguments for detecting foreground.
        robust_invert_kwargs : Dict
            Keyword arguments for robust inversion.

        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            The detection and contours arrays.
        """
        if (
            detect_foreground_kwargs["channel_axis"]
            != robust_invert_kwargs["channel_axis"]
        ):
            raise ValueError(
                "Channel axis must be the same for both detect_foreground "
                "and robust_invert."
            )

        channel = detect_foreground_kwargs["channel_axis"]

        shape = list(image.shape)
        if channel:
            shape.pop(channel + 1)

        detection, contours = _create_temp_detection_and_contours(shape)

        array_apply(
            image,
            out_array=contours,
            func=on_gpu(robust_invert),
            **robust_invert_kwargs,
        )

        array_apply(
            image,
            out_array=detection,
            func=on_gpu(detect_foreground),
            **detect_foreground_kwargs,
        )

        return detection, contours

    def _auto_from_labels(
        self, labels: ArrayLike, label_to_contours_kwargs: Dict
    ) -> None:
        """
        Automatically create contours from input labels.

        Parameters
        ----------
        labels : ArrayLike
            The input labels.
        label_to_contours_kwargs : Dict
            Keyword arguments for converting labels to contours.

        Returns
        -------
        Tuple[ArrayLike, ArrayLike]
            The detection and contours arrays.
        """
        store_detection = zarr.TempStore(suffix="detection")
        store_contours = zarr.TempStore(suffix="contours")

        labels_to_contours(
            labels,
            detection_store_or_path=store_detection,
            contours_store_or_path=store_contours,
            **label_to_contours_kwargs,
        )

        zarr_detection = zarr.open(store_detection)
        zarr_contours = zarr.open(store_contours)

        return zarr_detection, zarr_contours

    def _run_segmentation(self, detection: Layer, contours: Layer) -> None:
        """
        Run the segmentation step.

        Parameters
        ----------
        detection : napari.layers.Layer
            The detection napari layer.
        contours : napari.layers.Layer
            The contours napari layer.
        """
        segment(
            detection=detection.data,
            edge=contours.data,
            config=self.config,
            overwrite=True,
        )

    def _run_linking(self, scale: Sequence[float]) -> None:
        """
        Run the linking step.

        Parameters
        ----------
        scale : Sequence[float]
            The scale of the input data.
        """
        link(config=self.config, overwrite=True, scale=scale)

    def _run_solving(self) -> None:
        """
        Run the solving step.
        """
        solve(config=self.config, overwrite=True)

    def inputs_from_choice(self, choice: WorkflowChoice) -> List[UltrackInput]:
        """
        Get the required inputs based on the workflow choice.

        Parameters
        ----------
        choice : str
            The workflow choice.

        Returns
        -------
        List[str]
            A list of required inputs.
        """
        if choice == WorkflowChoice.AUTO_DETECT:
            return [UltrackInput.IMAGE]
        elif choice == WorkflowChoice.AUTO_FROM_LABELS:
            return [UltrackInput.LABELS]
        elif choice == WorkflowChoice.MANUAL:
            return [UltrackInput.DETECTION, UltrackInput.CONTOURS]
        else:
            raise ValueError(f"Unknown choice {choice}")


def _create_temp_detection_and_contours(
    shape: tuple[int, ...]
) -> tuple[ArrayLike, ArrayLike]:
    """
    Create temporary storage for detection and contours arrays.

    Parameters
    ----------
    shape : Tuple[int, ...]
        The shape of the arrays.

    Returns
    -------
    Tuple[ArrayLike, ArrayLike]
        The paths to the temporary detection and contours arrays.
    """
    detection_store = zarr.TempStore(suffix="detection")
    contours_store = zarr.TempStore(suffix="contours")

    zarr_detection = create_zarr(
        shape=shape,
        dtype=bool,
        store_or_path=detection_store,
        overwrite=True,
    )
    zarr_contours = create_zarr(
        shape=shape,
        dtype=np.float16,
        store_or_path=contours_store,
        overwrite=True,
    )

    return zarr_detection, zarr_contours
