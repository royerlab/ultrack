import asyncio
import json
import logging
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import zarr
from fastapi import FastAPI, HTTPException, WebSocket
from napari.components import ViewerModel
from napari.plugins import _initialize_plugins
from networkx.readwrite import json_graph
from numpy._typing import ArrayLike
from starlette.middleware.cors import CORSMiddleware
from starlette.responses import JSONResponse
from starlette.websockets import WebSocketState

import ultrack
from ultrack import link, segment, solve, to_tracks_layer, tracks_to_zarr
from ultrack.api.database import (
    Experiment,
    ExperimentStatus,
    create_experiment_instance,
    get_experiment,
    update_experiment,
)
from ultrack.api.settings import settings
from ultrack.api.utils.api import UltrackWebsocketLogger, raise_error
from ultrack.api.utils.fs import open_image
from ultrack.api.utils.zarr import get_channels_from_ome_zarr
from ultrack.config import MainConfig
from ultrack.core.export import tracks_layer_to_networkx, tracks_layer_to_trackmate
from ultrack.imgproc import detect_foreground, robust_invert
from ultrack.utils import labels_to_contours
from ultrack.utils.array import array_apply, create_zarr
from ultrack.utils.cuda import on_gpu

LOG = logging.getLogger(__name__)


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["null", "*", "http://localhost:3000", "http://localhost:8080"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

app.state.queue = asyncio.Queue()


@app.exception_handler(Exception)
async def exception_handler(_, exc: Exception) -> JSONResponse:
    """General FastAPI exception handler.

    It captures all untreated exceptions and avoid the server to crash, returning a
    JSONResponse with the exception details. It also removes the experiment from the
    execution queue if it is still there.

    Parameters
    ----------
    _ : Request
        The request that raised the exception. it is not used.

    exc : Exception
        The exception raised.

    Returns
    -------
    JSONResponse
        A JSONResponse with the exception details.
    """
    LOG.error(exc)
    if app.state.queue.qsize() > 0:
        app.state.queue.task_done()
    return JSONResponse(
        status_code=exc.status_code,
        content={"detail": f"HTTPException: {exc.detail}"},
    )


def read_images_from_experiment(
    experiment: Experiment, image_names: List[str]
) -> dict[str, ArrayLike]:
    image_mapping = {
        "labels": experiment.labels_channel_or_path,
        "detection": experiment.detection_channel_or_path,
        "edges": experiment.edges_channel_or_path,
        "image": experiment.image_channel_or_path,
    }
    selected = [(im_name, image_mapping[im_name]) for im_name in image_names]

    if experiment.data_url is not None:
        node = open_image(experiment.data_url)
        named_data = get_channels_from_ome_zarr(node, valid_channels=selected)
    else:
        _initialize_plugins()
        viewer = ViewerModel()
        named_data = {
            name: viewer.open(path=path, plugin="napari")[0].data
            for name, path in selected
        }
    return named_data


async def start_experiment(ws: WebSocket, exp: Experiment) -> None:
    """Start an experiment.

    It persists the experiment instance to the database, updates its status to QUEUED,
    waits for the queue to be available, puts the experiment in the queue, and sends
    the experiment details to the client through the WebSocket.

    Parameters
    ----------
    ws : WebSocket
        The WebSocket instance used to communicate with the client.
    exp : Experiment
        The experiment instance with the ultrack configurations to be started.

    See Also
    --------
    finish_experiment : Finish the started experiment.
    """
    create_experiment_instance(exp)
    exp.status = ExperimentStatus.QUEUED
    update_experiment(exp)
    await app.state.queue.join()
    await app.state.queue.put(exp)
    await ws.send_json(json.loads(exp.json()))
    UltrackWebsocketLogger.register_interruption_handler(exp.id)
    exp.status = ExperimentStatus.INITIALIZING
    update_experiment(exp)


async def finish_experiment(ws: WebSocket, exp: Experiment) -> Experiment:
    """Finish an experiment.

    Updates the experiment end time, and sends the final experiment state to the client
    through the WebSocket if it is still connected. It also removes the experiment from
    the execution queue.

    Parameters
    ----------
    ws : WebSocket
        The WebSocket instance used to communicate with the client.
    exp : Experiment
        The experiment instance to be finished.

    """
    exp.end_time = datetime.now()
    exp.status = ExperimentStatus.SUCCESS
    update_experiment(exp)
    app.state.queue.task_done()
    if ws.client_state == WebSocketState.CONNECTED:
        # send the final state
        await ws.send_json(json.loads(exp.json()))
    return exp


@app.get("/")
async def root() -> Dict[str, str]:
    """Root endpoint.

    Returns
    -------
    Dict[str, str]
        A dictionary with the ultrack version.
    """
    return {"ultrack.__version__": ultrack.__version__}


@app.get("/config/default")
async def get_default_config() -> Dict:
    """Gets the default ultrack configuration.

    Returns
    -------
    MainConfig
        The default ultrack configuration.
    """
    config = MainConfig()
    config.data_config = None
    return config.dict()


@app.get("/config/available")
async def get_available_configs() -> Dict:
    """Gets the default ultrack configuration.

    Returns
    -------
    MainConfig
        The default ultrack configuration.
    """
    default_config = MainConfig()
    default_config.data_config = None

    experiment = {
        "name": "Unnamed Experiment",
        "config": default_config.dict(),
    }

    auto_detect_config = {
        "experiment": experiment.copy(),
        "detect_foreground_kwargs": {},
        "robust_invert_kwargs": {},
    }
    auto_detect_config["experiment"]["image_channel_or_path"] = "..."

    manual_segment_config = {
        "experiment": experiment.copy(),
    }
    manual_segment_config["experiment"]["edges_channel_or_path"] = "..."
    manual_segment_config["experiment"]["detection_channel_or_path"] = "..."

    labels_to_edges_config = {
        "experiment": experiment.copy(),
        "label_to_edges_kwargs": {},
    }
    labels_to_edges_config["experiment"]["labels_channel_or_path"] = "..."

    return {
        "auto_detect": {
            "link": "/segment/auto_detect",
            "human_name": "Auto Detect Foreground and Edges From Image",
            "config": auto_detect_config,
        },
        "manual_segment": {
            "link": "/segment/manual",
            "config": manual_segment_config,
            "human_name": "Manual Input of Foreground and Edges",
        },
        "labels_to_edges": {
            "link": "/segment/labels",
            "config": labels_to_edges_config,
            "human_name": "Auto Detect Foreground and Edges from Labels",
        },
    }


@app.post("/stop/{experiment_id}")
async def request_experiment_interruption(experiment_id: int) -> Dict[str, str]:
    """Request the interruption of an experiment given its id.

    Parameters
    ----------
    experiment_id : int
        The id of the experiment to be interrupted.

    Returns
    -------
    Dict[str, str]
        A dictionary with the status of the interruption request.
    """
    UltrackWebsocketLogger.request_interruption(experiment_id)
    return {"status": "ok"}


@app.websocket("/segment/auto_detect")
async def auto_detect(websocket: WebSocket) -> None:
    """Detect automatically foreground and boundaries by simple image processing.

    The experiment is only started if the WebSocket is connected and the experiment is
    provided. See the Notes section for the expected JSON structure.


    Parameters
    ----------
    websocket : WebSocket
        The WebSocket instance used to communicate with the client.

    Notes
    -----
    The socket should receive a JSON with the following structure:
    {
        "experiment": {
            "data_url": "path/to/image.ome.zarr",
            "name": "Experiment Name",
            "image_channel_or_path": "ome_zarr channel_name",
            config: {
                ...
            },
        },
        "detect_foreground_kwargs": {
            ...
        },
        "robust_invert_kwargs": {
            ...
        }
    }
    where "detect_foreground_kwargs" and "robust_invert_kwargs" are optional and are
    the keyword arguments to be passed to the detect_foreground and robust_invert
    functions, respectively. experiment.config is a dictionary with the ultrack
    configuration as described in `ultrack.config.config.MainConfig`.

    See Also
    --------
    ultrack.imgproc.segmentation.detect_foreground : Automatically detects foreground
        given an image.
    ultrack.imgproc.segmentation.robust_invert : Automatically detects edges given an
        image.
    ultrack.api.database.Experiment : Experiment class definition.
    ultrack.config.config.MainConfig : ultrack configuration class definition.
    """
    await websocket.accept()
    data = await websocket.receive_json()

    if "experiment" not in data:
        await raise_error(
            ValueError("Experiment not provided. Please provide an experiment."),
            ws=websocket,
        )
        return

    experiment = Experiment.parse_obj(data["experiment"])

    try:
        detect_foreground_kwargs = data["detect_foreground_kwargs"]
    except KeyError:
        detect_foreground_kwargs = {}
        LOG.warning("detect_foreground_kwargs not provided. Using default values.")

    try:
        robust_invert_kwargs = data["robust_invert_kwargs"]
    except KeyError:
        robust_invert_kwargs = {}
        LOG.warning("robust_invert_kwargs not provided. Using default values.")

    async with UltrackWebsocketLogger(websocket, experiment):
        await start_experiment(websocket, experiment)

        named_data = read_images_from_experiment(experiment, image_names=["image"])

        image_data: zarr.Array = named_data["image"]

        shape = image_data.shape

        with tempfile.TemporaryDirectory() as temp_path:
            detection_path = str(Path(temp_path) / "detection.zarr")
            edges_path = str(Path(temp_path) / "edges.zarr")

            zarr_detection = create_zarr(
                shape=shape,
                dtype=float,
                store_or_path=detection_path,
                overwrite=True,
            )
            array_apply(
                image_data.astype("float32"),
                out_array=zarr_detection,
                func=on_gpu(detect_foreground),
                **detect_foreground_kwargs,
            )

            zarr_edges = create_zarr(
                shape=shape,
                dtype=float,
                store_or_path=edges_path,
                overwrite=True,
            )
            array_apply(
                image_data.astype("float32"),
                out_array=zarr_edges,
                func=on_gpu(robust_invert),
                **robust_invert_kwargs,
            )

            experiment.data_url = None
            experiment.image_channel_or_path = None
            experiment.labels_channel_or_path = None
            experiment.edges_channel_or_path = edges_path
            experiment.detection_channel_or_path = detection_path
            experiment.status = ExperimentStatus.DATA_LOADED

            update_experiment(experiment)

            await segment_link_and_solve(experiment, ws=websocket)

    await finish_experiment(websocket, experiment)


@app.websocket("/segment/manual")
async def manual_segment(websocket: WebSocket) -> None:
    """Manually provide the foreground and edges, subsequently segmenting and solving
    the experiment by ultrack.

    The experiment is only started if the WebSocket is connected and the experiment is
    provided. See the Notes section for the expected JSON structure.

    Parameters
    ----------
    websocket : WebSocket
        The WebSocket instance used to communicate with the client.

    Notes
    -----
    The socket should receive a JSON with the following structure:
    {
        "experiment": {
            "data_url": "path/to/image.ome.zarr",
            "name": "Experiment Name",
            "detection_channel": "ome_zarr_channel_name",
            "edges_channel": "ome_zarr_edges_channel_name",
            config: {
                ...
            },
        }
    }
    experiment.config is a dictionary with the ultrack
    configuration as described in `ultrack.config.config.MainConfig`.

    See Also
    --------
    ultrack.api.database.Experiment : Experiment class definition.
    ultrack.config.config.MainConfig : ultrack configuration class definition.
    """
    await websocket.accept()
    data = await websocket.receive_json()

    if "experiment" not in data:
        await raise_error(
            ValueError("Experiment not provided. Please provide an experiment."),
            ws=websocket,
        )
        return

    if isinstance(data["experiment"], str):
        experiment = json.loads(data["experiment"])
    else:
        experiment = data["experiment"]
    experiment = Experiment.parse_obj(experiment)

    async with UltrackWebsocketLogger(websocket, experiment):
        await start_experiment(websocket, experiment)

        experiment.status = ExperimentStatus.DATA_LOADED
        update_experiment(experiment)

        await segment_link_and_solve(experiment, websocket)

    await finish_experiment(websocket, experiment)


@app.websocket("/segment/labels")
async def auto_from_labels(websocket: WebSocket) -> None:
    """Detect automatically foreground and boundaries by simple image processing from
    an external label map provided by an instance segmentation algorithm or model.

    The experiment is only started if the WebSocket is connected and the experiment is
    provided. See the Notes section for the expected JSON structure.

    Parameters
    ----------
    websocket : WebSocket
        The WebSocket instance used to communicate with the client.

    Notes
    -----
    The socket should receive a JSON with the following structure:
    {
        "experiment": {
            "data_url": "path/to/image.ome.zarr",
            "name": "Experiment Name",
            "label_channel": "ome_zarr_label_channel_name",
            config: {
                ...
            },
        },
        "label_to_edges_kwargs": {
            ...
        },
    }
    where "label_to_edges_kwargs" is optional and is the keyword arguments to be passed
    to the labels_to_edges function. experiment.config is a dictionary with the ultrack
    configuration as described in `ultrack.config.config.MainConfig`.

    See Also
    --------
    ultrack.imgproc.segmentation.labels_to_edges : Automatically detects foreground
        and edges given a label map.
    ultrack.api.database.Experiment : Experiment class definition.
    ultrack.config.config.MainConfig : ultrack configuration class definition.
    """
    await websocket.accept()
    data = await websocket.receive_json()

    if "experiment" not in data:
        await raise_error(
            ValueError("Experiment not provided. Please provide an experiment."),
            ws=websocket,
        )
        return

    experiment = Experiment.parse_obj(data["experiment"])

    try:
        label_to_edges_kwargs = data["label_to_edges_kwargs"]
    except KeyError:
        label_to_edges_kwargs = {}
        LOG.warning("label_to_edges_kwargs not provided. Using default values.")

    async with UltrackWebsocketLogger(websocket, experiment):
        await start_experiment(websocket, experiment)

        named_data = read_images_from_experiment(experiment, image_names=["labels"])

        label_data: zarr.Array = named_data["labels"]

        with tempfile.TemporaryDirectory() as temp_path:
            detection_path = str(Path(temp_path) / "detection.zarr")
            edges_path = str(Path(temp_path) / "edges.zarr")

            labels_to_contours(
                label_data,
                foreground_store_or_path=detection_path,
                contours_store_or_path=edges_path,
                **label_to_edges_kwargs,
            )

            experiment.data_url = None
            experiment.image_channel_or_path = None
            experiment.labels_channel_or_path = None
            experiment.edges_channel_or_path = edges_path
            experiment.detection_channel_or_path = detection_path
            experiment.status = ExperimentStatus.DATA_LOADED

            update_experiment(experiment)

            await segment_link_and_solve(experiment, websocket)

    await finish_experiment(websocket, experiment)


async def segment_link_and_solve(experiment: Experiment, ws: WebSocket) -> None:
    """Segment, link, and solve an experiment with Ultrack.

    Parameters
    ----------
    experiment : Experiment
        The experiment to be executed. It should have the data_url, detection_channel,
        and edges_channel set.

    ws : WebSocket
        The WebSocket instance used to communicate with the client.

    Raises
    ------
    ProcessedException
        If any error occurs during the execution of the experiment.
    """
    try:
        named_data = read_images_from_experiment(
            experiment, image_names=["detection", "edges"]
        )

        detection = named_data["detection"]
        edges = named_data["edges"]

        config = experiment.get_config()
        experiment.status = ExperimentStatus.SEGMENTING
        update_experiment(experiment)
        segment(
            foreground=detection,
            contours=edges,
            config=config,
        )
    except Exception as e:
        await raise_error(e, experiment, ws)

    experiment.status = ExperimentStatus.LINKING
    update_experiment(experiment)
    link(config)

    experiment.status = ExperimentStatus.SOLVING
    update_experiment(experiment)
    solve(config)

    try:
        experiment.status = ExperimentStatus.EXPORTING
        update_experiment(experiment)
        tracks_df, graph = to_tracks_layer(config)
        G = tracks_layer_to_networkx(tracks_df)
        G = json_graph.node_link_data(G)
        experiment.tracks = json.dumps(G)
        segments_path = (
            settings.api_results_path / f"segments_id_exp_{experiment.id:06d}.zarr"
        )
        tracks_to_zarr(
            config,
            tracks_df,
            store_or_path=segments_path,
            overwrite=True,
        )
        experiment.final_segments_url = str(segments_path.absolute())
    except Exception as e:
        await raise_error(
            e, experiment, ws, f"Could not export tracks to zarr. {str(e)}"
        )


@app.get("/experiment/{experiment_id}/trackmate")
async def get_trackmate(experiment_id: int) -> Dict:
    """Get the TrackMate xml from an experiment.

    Parameters
    ----------
    experiment_id : int
        The id of the experiment to get the TrackMate xml.

    Returns
    -------
    Dict
        A dictionary with the TrackMate XML file.
    """
    experiment = get_experiment(experiment_id)
    if experiment is None:
        raise HTTPException(
            status_code=404, detail=f"Experiment {experiment_id} not found."
        )
    tracks_df, _ = to_tracks_layer(experiment.get_config())
    trackmate_xml = tracks_layer_to_trackmate(tracks_df)
    return {"trackmate_xml": trackmate_xml}


@app.websocket("/segment/model/{model_name}")
async def segment_by_model(websocket: WebSocket) -> None:
    """TODO: Segment by a model.

    Available models: cellpose, stardist2d, PlantSeg, and MicroSAM.

    Parameters
    -----------
    websocket : WebSocket
        The WebSocket instance used to communicate with the client.
    """
    raise HTTPException(status_code=501)


async def add_flow(experiment: Experiment, websocket: WebSocket) -> None:
    """TODO: Add flow to an experiment.

    Parameters
    -----------
    experiment : Experiment
        The experiment to be executed.
    websocket : WebSocket
        The WebSocket instance used to communicate with the client.
    """
    raise HTTPException(status_code=501)


async def auto_flow(experiment: Experiment, websocket: WebSocket) -> None:
    """TODO: Automatically compute the flow of an experiment.

    Parameters
    -----------
    experiment : Experiment
        The experiment to be executed.
    websocket : WebSocket
        The WebSocket instance used to communicate with the client.
    """
    raise HTTPException(status_code=501)
