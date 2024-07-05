REST API
--------

The ultrack REST API is a set of HTTP endpoints that allow you to track your data from a Ultrack server.
This is what enables the :doc:`Ultrack FIJI plugin <fiji>`.

.. autopydantic_model:: ultrack.api.database.Experiment

.. describe:: GET /config/available

    This endpoint returns the available configuration for the Ultrack server.

    .. code-block::json
            {
                "auto_detect": {
                    "link": "/segment/auto_detect",
                    "human_name": "Auto Detect Foreground and Edges From Image",
                    "config": {
                        "experiment": {
                            "name": "Experiment Name",
                            "config": MainConfig()
                        },
                        "detect_foreground_kwargs": {},
                        "robust_invert_kwargs": {},
                    }
                },
                ...
            }

    :return: The available configuration.


.. describe:: WEBSOCKET /segment/auto_detect

    This endpoint is a websocket endpoint that allows you to send a frame to the Ultrack server and get the segmentation result.

    This endpoint requires a JSON payload with the following structure:

    .. code-block::json
        {
            "experiment": {
                "name": "Experiment Name",
                "config": MainConfig()
            },
            "detect_foreground_kwargs": {},
            "robust_invert_kwargs": {},
        }

    and repeatedly sends the JSON payload from the server:

    .. code-block::json
        {
            "frame": frame,
            "segmentation": segmentation,
            "labels": labels,
            "experiment": {
                "name": "Experiment Name",
                "config": MainConfig()
            }
        }

    :param frame: The frame to segment.
    :type frame: numpy.ndarray

    :return: The segmentation result.
    :rtype: dict

.. describe:: WEBSOCKET /segment/manual

.. describe:: WEBSOCKET /segment/labels

.. describe:: GET /experiment/{experiment_id}/trackmate

