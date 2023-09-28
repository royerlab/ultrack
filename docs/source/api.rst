API
---

Core functionalities

.. autosummary::

    ultrack.track
    ultrack.segment
    ultrack.link
    ultrack.solve
    ultrack.add_flow
    ultrack.load_config

.. could not make it work for ultrack.utils.array

Image processing utilities

.. autosummary::

    ultrack.imgproc.segmentation.Cellpose
    ultrack.imgproc.plantseg.PlantSeg
    ultrack.imgproc.sam
    ultrack.imgproc.flow

Exporting

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

==========================
Image processing utilities
==========================

-------------------
DL models interface
-------------------

.. autoclass:: ultrack.imgproc.segmentation.Cellpose
    :members:

.. autoclass:: ultrack.imgproc.plantseg.PlantSeg
    :members:

.. automodule:: ultrack.imgproc.sam
    :members:

----
Flow
----

.. automodule:: ultrack.imgproc.flow
    :members:

================
Tracks utilities
================

.. automodule:: ultrack.tracks
    :imported-members:
    :members:

=========
Exporting
=========

.. automodule:: ultrack.core.export
    :members:
    :imported-members:
