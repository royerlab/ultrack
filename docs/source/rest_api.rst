REST API
========

The ultrack REST API is a set of HTTP/Websockets endpoints that allow you to track your
data from a Ultrack server.
This is what enables the :doc:`Ultrack FIJI plugin <fiji>`.

The communication between the Ultrack server and the client is mainly done through websockets.
This allows for a more efficient communication between the server and the client, enabling
real time responses.

All the messages sent through the websocket are JSON messages. And there is always an
:class:`Experiment <ultrack.api.database.Experiment>` object that is sent encoded within the message.
This object contains all the information about the experiment that is being run, including
the configuration (:class:`MainConfig <ultrack.config.MainConfig>`) of the experiment, the status of the
experiment (:enum:`ExperimentStatus <ultrack.api.database.ExperimentStatus>`), the experiment ID, and the experiment name.
When the experiment is concluded, this object will also contain the results of the
experiment, encoded in the fields ``final_segments_url`` (URL to the tracked segments path)
and ``tracks`` (JSON of napari complaint tracking format).

**One important notice is that the server is meant to be executed in an environment where
the client has access to the data that is being processed. This means that the server
does not store the data that is being processed, only the results of the experiment.
This also means that the server must have access to the data that is being processed,
so the client must send the links to the data that is being processed.**

Endpoints
---------

In the following sections, we will describe the available endpoints and the expected
payloads for each endpoint.

Meta endpoint
^^^^^^^^^^^^^

To avoid keeping track of each endpoint, there is a single endpoint that returns the available
endpoints for the Ultrack server. This also allows for the Ultrack server to be more dynamic,
as the available endpoints can be changed without changing the client. This endpoint is
describe below.

.. describe:: GET /config/available

    This endpoint returns all the available endpoints for the Ultrack server.
    The response is a JSON object with the following structure:

    .. code-block:: JSON

        {
            "id_endpoint": {
                "link": "/url/to/endpoint",
                "human_name": "The title of the endpoint",
                "config": {
                    "experiment": {
                        "name": "Experiment Name",
                        "config": "MainConfig()"
                    },
                    "set_of_kwargs_1": {},
                    "set_of_kwargs_2": {},
                    "...",
                    "set_of_kwargs_n": {}
                }
            },
            "..."
        }

    As you can see, the response is a JSON object with the keys being the endpoint ID
    and the values being a JSON object with the keys `link`, `human_name`, and `config`.
    The `link` key is the URL to the endpoint, the `human_name` key is the title of the endpoint,
    and the `config` key is the expected input payload for the endpoint.

    The `config` key comprises the initial configuration of the experiment
    (an instance of :class:`Experiment <ultrack.api.database.Experiment>`), and a
    possible set of keyword arguments that are expected by the endpoint. Those keyword
    arguments are dependent on the endpoint and are described in the following sections.

    The `experiment` instance will be initialized with the default configuration of the
    :class:`MainConfig <ultrack.config.MainConfig>` class. This configuration can be
    changed by the client as his needs and then be sent to the server.

Experiment endpoints
^^^^^^^^^^^^^^^^^^^^

The experiment endpoints are the main endpoints of the Ultrack server. They are the endpoints
that allow the client to run the experiments and get the results of the experiments.

.. describe:: WEBSOCKET /segment/auto_detect

    This endpoint is a websocket endpoint that allows you to send an image (referenced
    as ``image_channel_or_path``) to the server and get the segmentation of the image.

    This endpoint wraps the :func:`ultrack.imgproc.detect_foreground` function and the
    :func:`ultrack.imgproc.robust_invert` function, which are functions capable
    of obtaining the foreground of the image and its edges by image processing techniques.
    For that reason, one can override the default parameters of those functions by sending
    the ``detect_foreground_kwargs`` and ``robust_invert_kwargs`` as keyword arguments.
    Those keyword arguments will be passed to the respective functions.

    This endpoint requires a JSON payload with the following structure:

    .. code-block:: JSON

        {
            "experiment": {
                "name": "Experiment Name",
                "config": "..."
                "image_channel_or_path": "/path/to/image",
            },
            "detect_foreground_kwargs": {},
            "robust_invert_kwargs": {},
        }

    and repeatedly sends the :class:`Experiment <ultrack.api.database.Experiment>`
    JSON payload from the server. For example, the server could possibly send the
    following JSON payload:

    .. code-block:: JSON

        {
            "id": 1,
            "name": "Experiment Name",
            "status": "segmenting",
            "config": {
                "..."
            }
            "start_time": "2021-01-01T00:00:00",
            "end_time": "",
            "std_log": "Segmenting frame 1...",
            "err_log": "",
            "data_url": "",
            "image_channel_or_path": "/path/to/image",
            "edges_channel_or_path": "",
            "detection_channel_or_path": "",
            "segmentation_channel_or_path": "",
            "labels_channel_or_path": "",
            "final_segments_url": "",
            "tracks": ""
        }

    Alternatively, if the image is a OME-ZARR file, the input data could be referenced
    as a channel in the file. For example, the input data could be referenced as:

    .. code-block:: JSON

        {
            "experiment": {
                "name": "Experiment Name",
                "config": "..."
                "data_url": "/path/to/image.ome.zarr",
                "image_channel_or_path": "image_channel",
            },
            "detect_foreground_kwargs": {},
            "robust_invert_kwargs": {},
        }

.. describe:: WEBSOCKET /segment/manual

.. describe:: WEBSOCKET /segment/labels

.. describe:: GET /experiment/{experiment_id}/trackmate

Database API
^^^^^^^^^^^^

.. autopydantic_model:: ultrack.api.database.Experiment
    :members:

.. autoclass:: ultrack.api.database.ExperimentStatus