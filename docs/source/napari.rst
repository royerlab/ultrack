Napari plugin
-------------

We wrapped up most of the functionality in a napari widget. The widget is already installed
with the package, but you should have napari installed to use it. To use it, open
napari and select the widget from the plugins menu selecting ``ultrack`` and then ``Ultrack``
from the dropdown menu.

The plugin is built around the concept of a tracking workflow. Any workflow is a sequence
of pre-processing steps, segmentation, segments linking, and the tracking problem solver.
We explain the different workflows in the following sections.

Workflows
^^^^^^^^^

The difference between the workflows is the way the user provides the information to the plugin,
and the way the plugin processes the information. The remaining steps are the same for all workflows.
In that sense, ``segmentation``, ``linking``, and ``solver`` are the same for all workflows.
For each step, the widget provides direct access to the parameters of the step, and the user can
change the parameters to adapt the workflow to the specific problem. We explain how these
parameters behave in :doc:`Configuration docs <configuration>`, and, more specifically, in the
:class:`Experiment <ultrack.config.SegmentationConfig>`,
:class:`Linking <ultrack.config.LinkingConfig>`, and
:class:`Tracking <ultrack.config.TrackingConfig>` sections. Every input requested by the plugin
should be loaded beforehand as a layer in ``Napari``.

There are three workflows available in the plugin:

- **Automatic tracking from image**: This workflow is designed to track cells in a sequence of images.
  It uses simple image processing techniques to detect the cells and the contours of the cells. In this
  workflow, you can change the parameters of the image processing steps, and for that you can
  refer to the documentation of the functions used in the image processing steps:

    - :func:`ultrack.imgproc.detect_foreground`
    - :func:`ultrack.imgproc.robust_invert`

- **Manual tracking**: Since ultrack is designed to work with precomputed cell detection and
    contour detection, this workflow is designed to the situation where the user has already
    computed the cell detection and the contours of the cells. In this situation, no additional
    parameter is needed, you only need to provide the cell detection and the contours of the cells.

- **Automatic tracking from segmentation labels**: This workflow is designed to track cells
    in a sequence of images where the user has already computed the segmentation of the cells.
    The user can use the segmentation to track the cells in the sequence of images. This workflow
    wraps the function :meth:`ultrack.utils.labels_to_contours` to compute the contours of the cells
    from the segmentation, so please refer to the documentation of this function to understand
    its parameters required in the interface.

Flow Field Estimation
^^^^^^^^^^^^^^^^^^^^^

Additionally, in every workflow the user is able to use a flow field estimation to improve the tracking
results, a technique that estimates the movement of the cells in the sequence
of images. The flow field estimation is computed using the function :func:`ultrack.imgproc.flow.timelapse_flow`,
and the user can change its parameters in the interface. Please refer to the
documentation of the function to understand the required parameters.