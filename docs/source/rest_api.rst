REST API
========

The ultrack REST API is a set of HTTP/Websockets endpoints that allow you to track your
data from an Ultrack server.
This is what enables the :doc:`Ultrack FIJI plugin <fiji>`.

The communication between the Ultrack server and the client is mainly done through websockets.
Allowing real-time responses for efficient communication between the server and the client.

All the messages sent through the websocket are JSON messages. And there is always an
:class:`Experiment <ultrack.api.database.Experiment>` object encoded and sent within the message.
This object contains all the information about the experiment that is being run, including
the configuration (:class:`MainConfig <ultrack.config.MainConfig>`) of the experiment, the status of the
experiment (:class:`ExperimentStatus <ultrack.api.database.ExperimentStatus>`), the experiment ID, and the experiment name.
When the experiment is concluded, this object will also contain the results of the
experiment, encoded in the fields ``final_segments_url`` (URL to the tracked segments path)
and ``tracks`` (JSON of napari complaint tracking format).

**IMPORTANT:** The server must have access to the data shared by the client, for example, through the web, or a shared file system.
Because the server does not store the input data being processed, only the results of the experiment.

Endpoints
---------

In the following sections, we will describe the available endpoints and the expected
payloads for each endpoint.

Meta endpoint
^^^^^^^^^^^^^

To avoid keeping track of each endpoint, there is a single endpoint that returns the available
endpoints for the Ultrack server. This also allows for the Ultrack server to be more dynamic,
as the available endpoints can be changed without changing the client. This endpoint is
described below.

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

    The `experiment` instance is initialized with the default configuration of the
    :class:`MainConfig <ultrack.config.MainConfig>` class. This configuration can be
    changed by the client and sent to update the server.

Experiment endpoints
^^^^^^^^^^^^^^^^^^^^

The experiment endpoints are the main endpoints of the Ultrack server.
They allow the client to run the experiments and get their respective results.

.. _segment_auto_detect:
.. describe:: WEBSOCKET /segment/auto_detect

    This endpoint is a websocket endpoint that allows you to send an image (referenced
    as ``image_channel_or_path``) to the server and get the segmentation of the image.

    This endpoint wraps the :func:`ultrack.imgproc.detect_foreground` function and the
    :func:`ultrack.imgproc.robust_invert` function, which estimates
    the foreground of the image and its contours by image processing techniques.
    For that reason, one can override the default parameters of those functions by sending
    the ``detect_foreground_kwargs`` and ``robust_invert_kwargs`` as keyword arguments.
    Those keyword arguments will be passed to their respective function.

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

    and the reserver repeatedly returns the :class:`Experiment <ultrack.api.database.Experiment>`
    JSON payload. For example:

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

    Alternatively, if the image is an OME-ZARR file, the input data could be
    a specific channel. In this case, the input data could be referenced as:

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

All the other endpoints are similar to the :ref:`/segment/auto_detect <segment_auto_detect>` endpoint, but they
are more specific to the segmentation labels of the image. The endpoints are described below.

.. _segment_manual:
.. describe:: WEBSOCKET /segment/manual

    This endpoint is similar to the :ref:`/segment/auto_detect <segment_auto_detect>` endpoint, but it allows the
    client to manually provide cells' foreground mask and their multilevel contours.
    This endpoint requires the following JSON payload:

    .. code-block:: JSON

        {
            "experiment": {
                "name": "Experiment Name",
                "config": "..."
                "detection_channel_or_path": "/path/to/detection",
                "edges_channel_or_path": "/path/to/contours",
            },
        }

    Alternatively, if the image is an OME-ZARR file, the input data could be
    a specific channel. In this case, the input data could be referenced as:

    .. code-block:: JSON

        {
            "experiment": {
                "name": "Experiment Name",
                "config": "..."
                "data_url": "/path/to/image.ome.zarr",
                "detection_channel_or_path": "detection_channel",
                "edges_channel_or_path": "contours_channel",
            },
        }

    For both cases, the server will send the :class:`Experiment <ultrack.api.database.Experiment>`
    JSON payload. For example:

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
            "std_log": "Linking cells...",
            "err_log": "",
            "data_url": "",
            "image_channel_or_path": "",
            "edges_channel_or_path": "/path/to/contours",
            "detection_channel_or_path": "/path/to/detection",
            "segmentation_channel_or_path": "",
            "labels_channel_or_path": "",
            "final_segments_url": "",
            "tracks": ""
        }

Last but not least, the following endpoint could be used in a situation where the client already has
the instance segmentation of the cells, for example, from Cellpose or StarDist.

.. _segment_labels:
.. describe:: WEBSOCKET /segment/labels

    This endpoint is similar to the :ref:`/segment/auto_detect <segment_auto_detect>` endpoint, but it allows the
    client to provide pre-computed instance segmentation of the cells.
    This endpoint wraps the :meth:`ultrack.utils.labels_to_contours` function, which computes the foreground and contours
    of the cells from the instance segmentation.

    This endpoint requires the following JSON payload:

    .. code-block:: JSON

        {
            "experiment": {
                "name": "Experiment Name",
                "config": "..."
                "labels_channel_or_path": "/path/to/labels",
            },
            "labels_to_edges_kwargs": {},
        }

    Alternatively, if the image is an OME-ZARR file, the input data could be
    a specific channel. In this case, the input data could be referenced as:

    .. code-block:: JSON

        {
            "experiment": {
                "name": "Experiment Name",
                "config": "..."
                "data_url": "/path/to/image.ome.zarr",
                "labels_channel_or_path": "labels_channel",
            },
        }

    For both cases, the server will send the :class:`Experiment <ultrack.api.database.Experiment>`
    JSON payload. For example:

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
            "std_log": "Linking cells...",
            "err_log": "",
            "data_url": "",
            "image_channel_or_path": "",
            "edges_channel_or_path": "",
            "detection_channel_or_path": "",
            "segmentation_channel_or_path": "",
            "labels_channel_or_path": "/path/to/labels",
            "final_segments_url": "",
            "tracks": ""
        }

Data export endpoints
^^^^^^^^^^^^^^^^^^^^^

.. describe:: GET /experiment/{experiment_id}/trackmate

    This endpoint allows the client to download the results of the experiment in the TrackMate
    XML format. The client must provide the ``experiment_id`` in the URL. This id is obtained from
    the :class:`Experiment <ultrack.api.database.Experiment>` instance that was executed.
    The server will return an XML encoded within the response.

Database Schema
---------------

All the data that is being processed by the Ultrack server is stored in a database. This
database is a SQLite database that is created when the server is started. The database
is used to store the results of the experiments and the configuration of the experiments.

The database schema is the same as the one used in Ultrack, but with an additional table
to store the configuration and the status of the experiments. The schema is described below.

.. autopydantic_model:: ultrack.api.database.Experiment
    :members:

.. autoclass:: ultrack.api.database.ExperimentStatus
