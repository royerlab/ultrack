Getting started
---------------

Ultrack tracking pipeline is divided into three main steps:

- ``segment``: Creating the candidate segmentation hypotheses;
- ``link``: Finding candidate links between segmentation hypotheses of adjacent frames;
- ``solve``: Solving the tracking problem by finding the best segmentation and trajectory for each cell.

These three steps have their respective function with the same names and configurations but are summarized in the ``track`` function or the ``Tracker`` class, which are the main entry point for the tracking pipeline.

You'll notice that these functions do not return any results. Instead, they store the results in a database. This enables us to process datasets larger than memory, and distributed or parallel computing. We provide auxiliary functions to export the results to a format of your choice.

The ``MainConfig.data_config`` provides the interface to interact with the database, so beware of using ``overwrite`` parameter when re-executing these functions, to erase previous results otherwise it will build on top of existing data.

If you want to go deep into the weeds of our backend. We recommend looking at the ``ultrack.core.database.py`` file.

Each one of the main steps will be explained in detail below, a detailed description of the parameters can be found in :doc:`configuration`.

Segmentation
````````````

Ultrack's canonical inputs are a ``foreground`` and a ``contours``, there are several ways to obtain these inputs, which will be explained below. For now, let's consider we are working with them directly.

Both ``foreground`` and ``contours`` maps must have the same shape, with the first dimension being time (``T``) and the remaining being the spatial dimensions (``Z`` optional, and ``Y``, ``X``).

``foreground`` is used with ``config.segmentation_config.threshold`` to create a binary mask indicating the presence of the object of interest, it's by default 0.5. Values above the threshold are considered as foreground, and values below are considered as background.

``contours`` indicates the confidence of each pixel (voxel) being a cell boundary (contour). The higher the value, the more likely it is a cell boundary. It couples with ``config.segmentation_config.min_frontier`` which fuses segmentation candidates separated by a boundary with an average value below this threshold, it's by default 0, so no fusion is performed.

The segmentation is the most important step, as it will define the candidates for the tracking problem.
If your cells of interest are not present in the ``foreground`` after the threshold, you won't be able to track them.
If there isn't any faint boundary in the ``contours``, you won't be able to split into individual cells. That's why it's preferred to have a lot of contours (more hypotheses), even incorrect ones than having none.

Linking
```````

The linking step is responsible for finding candidate links between segmentation hypotheses of adjacent frames. Usually, not a lot of candidates are needed (``config.linking_config.max_neighbors = 5``), unless you have several segmentation hypotheses (``contours`` with several gray levels).

The parameter ``config.linking_config.max_distance`` must be at least the maximum distance between two cells in consecutive frames. It's used to filter out candidates that are too far from each other. If this value is too small, you won't be able to link cells that are far from each other.

Solving
```````

The solving step is responsible for solving the tracking problem by finding the best segmentation and trajectory for each cell. The parameters for this step are harder to interpret, as they are related to the optimization problem. The most important ones are:

- ``config.tracking_config.appear_weight``: The penalization for a cell to appear, which means to start a new lineage;
- ``config.tracking_config.division_weight``: The penalization for a cell to divide, breaking a single tracklet into two;
- ``config.tracking_config.disappear_weight``: The penalization for a cell to disappear, which means to end a lineage;

These weights are negative or zero, as they try to balance the cost of including new lineages in the final solution. The connections (links) between segmentation hypotheses are positive and measure the quality of the tracks, so only lineages with a total linking weight higher than the penalizations are included in the final solution. At the same time, our optimization problem is finding the combination of connections that maximize the sum of weights of all lineages.

See the :ref:`tracking configuration description <tracking_config>` for more information and :doc:`optimizing` for details on how to select these parameters.


Exporting
`````````

Once the above steps have been applied, the tracking solutions are recorded in the database and they can be exported to a format of your choice, them being, ``to_networkx``, ``to_trackmate``, ``to_tracks_layer``, ``tracks_to_zarr`` and others.

See the :ref:`export API reference <api_export>` for all available options and their parameters.

Example of exporting solutions to napari tracks layer:

.. code-block:: python

    # ... tracking computation

    # Exporting to napari format using `Tracker` class
    tracks, graph = tracker.to_tracks_layer()

    # Exporting using config file
    tracks, graph = to_tracks_layer(config)


Post-processing
```````````````

We also provide some additional post-processing functions, to remove, join, or analyze your tracks. Most of them are available in ``ultrack.tracks``. Some examples are:

- ``close_tracks_gaps``: That closes gaps by joining tracklets and interpolating the missing segments;
- ``filter_short_sibling_tracks``: That removes short tracklets generated by false divisions;
- ``get_subgraph``: Which returns the whole lineage(s) of a given tracklet.

Other functionalities can be found in ``ultrack.utils`` or ``ultrack.imgproc``, one notable example is:

- ``tracks_properties``: Which returns compute statistics from the tracks, segmentation masks and images.

For additional information, please refer to the :ref:`tracks post-processing API reference <api_tracks>`.

Image processing
````````````````

Despite being presented here last, ultrack's image processing module provides auxiliary functions to process your image before the segmentation step. It's not mandatory to use it, but it might reduce the amount of code you need to write to preprocess your images.

Most of them are available in ``ultrack.imgproc`` , ``ultrack.utils.array`` and ``ultrack.utils.cuda`` modules.

Refer to the :ref:`image processing API reference <api_imgproc>` for more information.
