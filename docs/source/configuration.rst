Configuration
-------------

The configuration is at the heart of ultrack, it is used to define the parameters for each step of the pipeline and where to store the intermediate results.
The `MainConfig` is the main configuration that contains the other configurations of the individual steps plus the data configuration.

The configurations are documented below, the parameters are ordered by importance, most important parameters are at the top of the list. Parameters that should not be changed in most of the cases are at the bottom of the list and contain a ``SPECIAL`` tag.

.. autosummary::

    ultrack.config.MainConfig
    ultrack.config.DataConfig
    ultrack.config.SegmentationConfig
    ultrack.config.LinkingConfig
    ultrack.config.TrackingConfig

---------------

.. autopydantic_model:: ultrack.config.MainConfig

---------------

.. autopydantic_model:: ultrack.config.DataConfig

---------------

.. autopydantic_model:: ultrack.config.SegmentationConfig

---------------

.. autopydantic_model:: ultrack.config.LinkingConfig

---------------

.. _tracking_config:

.. autopydantic_model:: ultrack.config.TrackingConfig
