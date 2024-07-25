API
===

First, we provide a summary of the main functionalities of the package.
Then we provide detailed documentation of every public function of ultrack.

Object Oriented API
^^^^^^^^^^^^^^^^^^^

.. autosummary::

    ultrack.Tracker

Core functionalities
^^^^^^^^^^^^^^^^^^^^

.. autosummary::

    ultrack.track
    ultrack.segment
    ultrack.link
    ultrack.solve
    ultrack.add_flow
    ultrack.load_config

.. could not make it work for ultrack.utils.array

Image processing utilities
^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autosummary::

    ultrack.imgproc.PlantSeg
    ultrack.imgproc.detect_foreground
    ultrack.imgproc.inverted_edt
    ultrack.imgproc.normalize
    ultrack.imgproc.robust_invert
    ultrack.imgproc.tracks_properties
    ultrack.imgproc.Cellpose
    ultrack.imgproc.sam.MicroSAM
    ultrack.imgproc.register_timelapse
    ultrack.imgproc.flow.timelapse_flow
    ultrack.utils.labels_to_contours

Exporting
^^^^^^^^^

.. autosummary::

   ultrack.core.export.to_ctc
   ultrack.core.export.to_trackmate
   ultrack.core.export.to_tracks_layer
   ultrack.core.export.tracks_to_zarr

====================
Core functionalities
====================

.. automodule:: ultrack
    :members:
    :imported-members:

===============
Array utilities
===============

.. automodule:: ultrack.utils.array
    :members:

.. _api_imgproc:

==========================
Image processing utilities
==========================

.. automodule:: ultrack.imgproc
    :members:
    :imported-members:

.. autofunction:: ultrack.utils.labels_to_contours

.. automodule:: ultrack.imgproc.sam
    :members:

.. _api_flow:

----
Flow
----

.. automodule:: ultrack.imgproc.flow
    :members:

.. _api_tracks:

================
Tracks utilities
================

.. automodule:: ultrack.tracks
    :imported-members:
    :members:

.. _api_export:

=========
Exporting
=========

.. automodule:: ultrack.core.export
    :members:
    :imported-members:
