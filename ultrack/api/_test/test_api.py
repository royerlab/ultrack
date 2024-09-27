import copy
import datetime
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest
import zarr
from fastapi.testclient import TestClient

import ultrack
from ultrack import MainConfig, link, segment, solve, to_tracks_layer, tracks_to_zarr
from ultrack.api.app import app
from ultrack.api.database import Experiment, ExperimentStatus
from ultrack.api.settings import settings
from ultrack.api.utils.fs import open_image
from ultrack.api.utils.zarr import get_channels_from_ome_zarr
from ultrack.core.database import clear_all_data
from ultrack.imgproc import detect_foreground, robust_invert
from ultrack.utils import labels_to_contours
from ultrack.utils.array import array_apply
from ultrack.utils.cuda import on_gpu


def _get_ultrack_solution(
    config: MainConfig, foreground: np.ndarray, edges: np.ndarray
) -> Tuple[pd.DataFrame, Dict, zarr.Array]:
    clear_all_data(settings.ultrack_data_config.database_path)

    segment(foreground=foreground, contours=edges, config=config)
    link(config)
    solve(config)

    tracks_df, graph = to_tracks_layer(config)
    segments = tracks_to_zarr(
        config,
        tracks_df,
        store_or_path=zarr.MemoryStore(),
        overwrite=True,
    )

    return tracks_df, graph, segments


@pytest.fixture
def ome_zarr_dataset_path(
    tmp_path: Path, timelapse_mock_data: Tuple[zarr.Array, zarr.Array, zarr.Array]
) -> str:
    from ome_zarr.io import parse_url
    from ome_zarr.writer import write_image

    path = str(tmp_path / "data.ome.zarr")
    detection, edges, labels = timelapse_mock_data
    data = np.stack([detection, detection, edges, labels], axis=1)

    store = parse_url(path, mode="w").store
    root = zarr.group(store=store)
    write_image(
        image=data, group=root, storage_options=dict(chunks=(1, *detection.chunks))
    )
    root.attrs["omero"] = {
        "channels": [{"label": v} for v in ["image", "detection", "edges", "labels"]]
    }

    return path


@pytest.fixture
def experiment_instance(
    ome_zarr_dataset_path: str,
    config_instance: MainConfig,
    tmp_path: Path,
) -> Experiment:
    settings.api_results_path = tmp_path

    experiment = Experiment(
        name="PyTest",
        config=config_instance,
        data_url=ome_zarr_dataset_path,
        image_channel_or_path="image",
        edges_channel_or_path="edges",
        detection_channel_or_path="detection",
        labels_channel_or_path="labels",
    )

    return experiment


def test_read_main():
    client = TestClient(app)
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"ultrack.__version__": ultrack.__version__}


def test_config():
    client = TestClient(app)
    response = client.get("/config/default")
    assert response.status_code == 200

    default_config = MainConfig()
    default_config.data_config = None

    assert response.json() == default_config.dict()


def test_manual_segment(experiment_instance: Experiment):
    experiment = copy.deepcopy(experiment_instance)
    # compare the results with the ones obtained from the ultrack module
    # must be done before processing
    node = open_image(experiment.data_url)
    named_data = get_channels_from_ome_zarr(
        node,
        valid_channels=[
            ("detection", experiment.detection_channel_or_path),
            ("edges", experiment.edges_channel_or_path),
        ],
    )
    detection = named_data[experiment.detection_channel_or_path]
    edges = named_data[experiment.edges_channel_or_path]

    tracks_df, graph, segments = _get_ultrack_solution(
        experiment.get_config(), detection, edges
    )

    # processing
    client = TestClient(app)
    with client.websocket_connect("/segment/manual") as websocket:
        json_exp = json.loads(
            json.dumps(
                experiment.dict(),
                default=lambda o: o.isoformat()
                if isinstance(o, (datetime.date, datetime.datetime))
                else None,
            )
        )
        websocket.send_json({"experiment": json_exp})
        while experiment.status != ExperimentStatus.SUCCESS:
            response = websocket.receive_json()
            experiment = Experiment.parse_obj(response)
            assert experiment.status != ExperimentStatus.ERROR

    tracks_df_api, graph_api = to_tracks_layer(experiment.get_config())
    segments_api = zarr.open(experiment.final_segments_url)

    assert tracks_df.equals(tracks_df_api)
    assert graph == graph_api
    assert np.array_equal(segments_api, segments)


@pytest.mark.parametrize(
    "detect_foreground_kwargs, robust_invert_kwargs",
    [(None, None), ({"sigma": 5}, {"voxel_size": [2, 2, 2]})],
)
def test_auto_detect(
    experiment_instance: Experiment,
    detect_foreground_kwargs: dict,
    robust_invert_kwargs: dict,
):
    experiment = copy.deepcopy(experiment_instance)

    # compare the results with the ones obtained from the ultrack module
    # must be done before processing
    node = open_image(experiment.data_url)
    named_data = get_channels_from_ome_zarr(
        node,
        valid_channels=[
            ("image", experiment.image_channel_or_path),
            ("edges", experiment.edges_channel_or_path),
            ("detection", experiment.detection_channel_or_path),
        ],
    )
    image_data = named_data[experiment.image_channel_or_path]

    # processing
    client = TestClient(app)
    with client.websocket_connect("/segment/auto_detect") as websocket:
        json_exp = json.loads(
            json.dumps(
                experiment.dict(),
                default=lambda o: o.isoformat()
                if isinstance(o, (datetime.date, datetime.datetime))
                else None,
            )
        )

        json_request = {"experiment": json_exp}

        if detect_foreground_kwargs:
            json_request["detect_foreground_kwargs"] = detect_foreground_kwargs
        else:
            detect_foreground_kwargs = {}

        if robust_invert_kwargs:
            json_request["robust_invert_kwargs"] = robust_invert_kwargs
        else:
            robust_invert_kwargs = {}

        websocket.send_json(json_request)
        while experiment.status != ExperimentStatus.SUCCESS:
            response = websocket.receive_json()
            experiment = Experiment.parse_obj(response)
            assert experiment.status != ExperimentStatus.ERROR

    detection = np.zeros_like(image_data, dtype=float)
    array_apply(
        image_data,
        out_array=detection,
        func=on_gpu(detect_foreground),
        **detect_foreground_kwargs,
    )

    edges = np.zeros_like(image_data, dtype=float)
    array_apply(
        image_data.astype(np.float32),
        out_array=edges,
        func=on_gpu(robust_invert),
        **robust_invert_kwargs,
    )
    tracks_df, graph, segments = _get_ultrack_solution(
        experiment.get_config(), detection, edges
    )

    tracks_df_api, graph_api = to_tracks_layer(experiment.get_config())
    segments_api = zarr.open(experiment.final_segments_url)

    assert tracks_df.equals(tracks_df_api)
    assert graph == graph_api
    assert np.array_equal(segments_api, segments)


@pytest.mark.parametrize(
    "label_to_edges_kwargs",
    [None, {"sigma": 5}],
)
def test_from_labels(experiment_instance: Experiment, label_to_edges_kwargs: dict):
    experiment = copy.deepcopy(experiment_instance)

    # must be loaded before processing
    node = open_image(experiment.data_url)
    named_data = get_channels_from_ome_zarr(
        node, valid_channels=[("labels", experiment.labels_channel_or_path)]
    )
    label_data = named_data[experiment.labels_channel_or_path]

    # processing
    client = TestClient(app)
    with client.websocket_connect("/segment/labels") as websocket:
        json_exp = json.loads(
            json.dumps(
                experiment.dict(),
                default=lambda o: o.isoformat()
                if isinstance(o, (datetime.date, datetime.datetime))
                else o,
            )
        )
        if label_to_edges_kwargs:
            websocket.send_json(
                {"experiment": json_exp, "label_to_edges_kwargs": label_to_edges_kwargs}
            )
        else:
            websocket.send_json({"experiment": json_exp})
            label_to_edges_kwargs = {}
        while experiment.status != ExperimentStatus.SUCCESS:
            response = websocket.receive_json()
            experiment = Experiment.parse_obj(response)
            assert experiment.status != ExperimentStatus.ERROR

    # compare the results with the ones obtained from the ultrack module
    detection, edges = labels_to_contours(
        label_data,
        foreground_store_or_path=zarr.MemoryStore(),
        contours_store_or_path=zarr.MemoryStore(),
        **label_to_edges_kwargs,
    )

    tracks_df, graph, segments = _get_ultrack_solution(
        experiment.get_config(), detection, edges
    )

    tracks_df_api, graph_api = to_tracks_layer(experiment.get_config())
    segments_api = zarr.open(experiment.final_segments_url)

    assert tracks_df.equals(tracks_df_api)
    assert graph == graph_api
    assert np.array_equal(segments_api, segments)


def test_output_experiment(experiment_instance: Experiment):
    client = TestClient(app)
    with client.websocket_connect("/segment/auto_detect") as websocket:
        json_exp = json.loads(
            json.dumps(
                experiment_instance.dict(),
                default=lambda o: o.isoformat()
                if isinstance(o, (datetime.date, datetime.datetime))
                else None,
            )
        )
        websocket.send_json({"experiment": json_exp})

        # skip the first message
        websocket.receive_json()

        while experiment_instance.status != ExperimentStatus.SUCCESS:
            response = websocket.receive_json()
            experiment_instance = Experiment.parse_obj(response)
            assert experiment_instance.status != ExperimentStatus.ERROR

        assert "robust_invert" in experiment_instance.err_log


def test_available_configs(experiment_instance: Experiment):
    client = TestClient(app)

    response = client.get("/config/available")
    assert response.status_code == 200

    for _, data in response.json().items():
        experiment = copy.deepcopy(experiment_instance)
        uri = data["link"]
        config_metadata = data["config"]
        config_metadata["experiment"]["data_url"] = experiment.data_url
        config_metadata["experiment"]["name"] = "PyTest"
        config_metadata["experiment"][
            "image_channel_or_path"
        ] = experiment.image_channel_or_path
        config_metadata["experiment"][
            "edges_channel_or_path"
        ] = experiment.edges_channel_or_path
        config_metadata["experiment"][
            "detection_channel_or_path"
        ] = experiment.detection_channel_or_path
        config_metadata["experiment"][
            "labels_channel_or_path"
        ] = experiment.labels_channel_or_path

        with client.websocket_connect(uri) as websocket:
            json_exp = json.loads(
                json.dumps(
                    config_metadata["experiment"],
                    default=lambda o: o.isoformat()
                    if isinstance(o, (datetime.date, datetime.datetime))
                    else None,
                )
            )
            websocket.send_json({"experiment": json_exp})

            # skip the first message
            websocket.receive_json()

            while experiment.status != ExperimentStatus.SUCCESS:
                response = websocket.receive_json()
                experiment = Experiment.parse_obj(response)
                assert experiment.status != ExperimentStatus.ERROR


# def test_exception_logging(experiment_instance: Experiment):
#     experiment_instance.labels_channel = "non_existent_channel"
#
#     client = TestClient(app)
#     with client.websocket_connect("/segment/labels") as websocket:
#         json_exp = json.loads(
#             json.dumps(
#                 experiment_instance.dict(),
#                 default=lambda o: o.isoformat()
#                 if isinstance(o, (datetime.date, datetime.datetime))
#                 else None,
#             )
#         )
#         websocket.send_json({"experiment": json_exp})
#
#         # skip the first message
#         websocket.receive_json()
#
#         while experiment_instance.status != ExperimentStatus.ERROR:
#             response = websocket.receive_json()
#             experiment_instance = Experiment.parse_obj(response)
#
#         assert "Exception" in experiment_instance.err_log
#         assert "Traceback" in experiment_instance.err_log
#         assert "non_existent_channel" in experiment_instance.err_log
#         assert "ValueError" in experiment_instance.err_log
#         assert experiment_instance.status == ExperimentStatus.ERROR


# def test_interruption(experiment_instance: Experiment):
#     client = TestClient(app)
#     with client.websocket_connect("/segment/manual") as websocket:
#         json_exp = json.loads(
#             json.dumps(
#                 experiment_instance.dict(),
#                 default=lambda o: o.isoformat()
#                 if isinstance(o, (datetime.date, datetime.datetime))
#                 else None,
#             )
#         )
#         websocket.send_json({"experiment": json_exp})
#         response = websocket.receive_json()
#         experiment_instance = Experiment.parse_obj(response)
#         response = client.post(f"/stop/{experiment_instance.id}")
#         assert response.status_code == 200
#         response = websocket.receive_json()
#         experiment_instance = Experiment.parse_obj(response)
#         assert experiment_instance.status == ExperimentStatus.ERROR
