Optimizing tracking performance
-------------------------------

Once you have a working ultrack pipeline, the next step is optimizing the tracking performance.
Here we describe our guidelines for optimizing the tracking performance and up to what point you can expect to improve the tracking performance.

It will be divided into a few sections:
- Pre-processing: How to make tracking easier by pre-processing the data;
- Input verification: Guidelines to check if you have good `labels` or `foreground` and `contours` maps;
- Hard constraints: Parameters must be adjusted so the hypotheses include the correct solution;
- Tracking tuning: Guidelines to adjust the weights to make the correct solution more likely.

Pre-processing
``````````````

The first question to ask yourself is, are your frames correctly aligned?
If not, we recommend aligning them, we provide the ``ultrack.imgproc.register_timelapse`` to align translations, see :doc:`API reference <api>`.

If your cells are very dynamic and there are considerable movements in different directions, we recommend using the ``flow`` functionalities to align individual segments with their own transforms, see the :doc:`flow tutorial <examples>`.

Some deep learning models are sensitive to the contrast of your data, we recommend adjusting the contrast and removing background applying them.

Input verification
`````````````````

At this point, we assume you already have a ``labels`` image or a ``foreground`` and ``contours`` maps;

You should check if ``labels`` or ``foreground`` contains every cell you want to track.
Any region that is not included in the ``labels`` or ``foreground`` will not be tracked and can only be fixed with post-processing.

If you are using ``foreground`` and ``contours`` maps, you should check if the contours induce hierarchies that lead to your desired segmentation.

This can be done by loading the ``contours`` in napari and viewing them over your original image with ``blending='additive'``.

You want your ``contours`` image to have higher values in the boundary of cells and lower values inside it.
This indicates that these regions are more likely to be boundaries than the interior of cells.
Notice, that this notion is much more flexible than a real contour map, which is we can use an intensity image as a `contours` map or an inverted distance transform.

In cells where this is not the case it is less likely ultrack will be able to separate them into individual segments.
However, optimizing contours is a complex task, I would continue to the next steps and look for specialized solutions once you have a working pipeline.
Some of these solutions are `PlantSeg <https://github.com/kreshuklab/plant-seg>`_ for membranes or `GoNuclear <https://github.com/kreshuklab/go-nuclear>`_ for nuclei.

Hard constraints
````````````````

This section is about adjusting the parameters so we have hypotheses that include the correct solution.

Please refer to the :doc:`Configuration docs <configuration>` as we refer to different parameters.

The expected cell size should be between ``segmentation_config.min_area`` and ``segmentation_config.max_area``.
Having a tight range assists in finding a good segmentation and significantly reduces the computation.
Our rule of thumb is to set the ``min_area`` to half the size of the expected cell or the smallest cell, *disregarding outliers*.
And the ``max_area`` to 1.25~1.5 the size of the largest cell, this is less problematic than the ``min_area``.

`linking_config.max_distance` should be set to the maximum distance a cell can move between frames.
We recommend setting some tolerance, for example, 1.5 times the expected movement.

Tracking tuning
```````````````

Once you have gone through the previous steps, you should have a working pipeline and now we can focus on the results and what can be done in each scenario.
