Quickstart
==========

This quickstart guide is recommended for users who are already familiar with Python and image analysis.
Otherwise, we recommend you read the :doc:`install` and :doc:`getting_started` sections.

Installation
------------

If already have a working Python environment, you can install ``ultrack`` using pip.
We recommend you use a conda environment to avoid any conflicts with your existing packages.
If you're using OSX or for additional information on how to create a conda environment and install packages, see :doc:`install`.

.. code-block:: bash

   pip install ultrack

Basic usage
-----------

The following example demonstrates how to use ``ultrack`` to track cells using its canonical input, a binary image of the foreground and a cells' contours image.

.. code-block:: python

   import napari
   from ultrack import MainConfig, Tracker

   # import to avoid multi-processing issues
   if __name__ == "__main__":

      # Load your data
      foreground = ...
      contours = ...

      # Create a config
      config = MainConfig()

      # Run the tracking
      tracker = Tracker(config=config)
      tracker.track(foreground=foreground, contours=contours)

      # Visualize the results
      tracks, graph = tracker.to_tracks_layer()
      napari.view_tracks(tracks[["track_id", "t", "y", "x"]], graph=graph)
      napari.run()


If you already have segmentation labels, you can provide them directly to the tracker.

.. code-block:: python

   import napari
   from ultrack import MainConfig, Tracker

   # import to avoid multi-processing issues
   if __name__ == "__main__":

      # Load your data
      labels = ...

      # Create a config
      config = MainConfig()

      # this removes irrelevant segments from the image
      # see the configuration section for more details
      config.segmentation_config.min_frontier = 0.5

      # Run the tracking
      tracker = Tracker(config=config)
      tracker.track(labels=labels)

      # Visualize the results
      tracks, graph = tracker.to_tracks_layer()
      napari.view_tracks(tracks[["track_id", "t", "y", "x"]], graph=graph)
      napari.run()
