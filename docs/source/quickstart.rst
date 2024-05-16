Quickstart
==========

Welcome to the quickstart guide for `ultrack`.

This guide will walk you through the initial steps to get you started with the package.

Installation
------------

To install `ultrack`, you can use pip:

.. code-block:: bash

   pip install ultrack

Basic Usage
-----------

.. code-block:: python

   import napari
   from ultrack import MainConfig, track, to_tracks_layer

   # Load your data
   foreground = ...
   boundaries = ...

   # Create a config
   config = MainConfig()

   # Run the tracking
   track(
      foreground=foreground,
      edges=boundaries,
      config=config,
   )

   # Visualize the results
   tracks, graph = to_tracks_layer(config)
   napari.view_tracks(
      tracks[["track_id", "t", "y", "x"]],
      graph=graph,
   )

Advanced Features
-----------------

See `examples <https://github.com/royerlab/ultrack/tree/main/examples>`_ for advanced features.

Contribute
----------

`ultrack` is an open-source project. We welcome and appreciate any contributions. For more details, check out our GitHub repository:

- `ultrack on GitHub <https://github.com/royerlab/ultrack>`_

Support and Feedback
--------------------

For support, issues, or feedback, please `open an issue <https://github.com/royerlab/ultrack/issues/new>`_ in the GitHub repository.

Citing
-------

If you use `ultrack` in your research, please cite the following paper:

.. code-block:: bibtex

   @article{bragantini2023ultrack,
      title={Large-Scale Multi-Hypotheses Cell Tracking Using Ultrametric Contours Maps},
      author={Bragantini, Jord{\~a}o and Lange, Merlin and Royer, Lo{\"\i}c},
      journal={arXiv preprint arXiv:2308.04526},
      year={2023}
   }

License
-------

BSD 3-Clause License
