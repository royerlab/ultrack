Tuning tracking performance
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

Registration
^^^^^^^^^^^^

Before tracking, the first question to ask yourself is, are your frames correctly aligned?

If not, we recommend aligning them. To do that, we provide the ``ultrack.imgproc.register_timelapse`` to align translations, see the :ref:`registration API <api_imgproc>`.

If the movement is more complex, with cells moving in different directions, we recommend using the ``flow`` functionalities to align individual segments with distinct transforms, see the :doc:`flow tutorial <examples>`.
See the :ref:`flow estimation API <api_flow>` for more information.

Deep learning
^^^^^^^^^^^^^

Some deep learning models are sensitive to the contrast of your data, we recommend adjusting the contrast and removing background before applying them to improve their predictions.
See the :ref:`image processing utilities API <api_imgproc>` for more information.

Input verification
``````````````````

At this point, we assume you already have a ``labels`` image or a ``foreground`` and ``contours`` maps;

You should check if ``labels`` or ``foreground`` contains every cell you want to track.
Any region that is not included in the ``labels`` or ``foreground`` will not be tracked and can only be fixed with post-processing.

If you are using ``foreground`` and ``contours`` maps, you should check if the contours induce hierarchies that lead to your desired segmentation.

This can be done by loading the ``contours`` in napari and viewing them over your original image with ``blending='additive'``.

You want your ``contours`` image to have higher values in the boundary of cells and lower values inside it.
This indicates that these regions are more likely to be boundaries than the interior of cells.
Notice, that this notion is much more flexible than a real contour map, which is we can use an intensity image as a `contours` map or an inverted distance transform.

In cells where this is not the case it is less likely ultrack will be able to separate them into individual segments.

If your cells (nuclei) are convex it is worth trying the ``ultrack.imgproc.inverted_edt`` for the ``contours``.

If even after going through the next steps you don't have successful results, I suggest looking for specialized solutions once you have a working pipeline.
Some of these solutions are `PlantSeg <https://github.com/kreshuklab/plant-seg>`_ for membranes or `GoNuclear <https://github.com/kreshuklab/go-nuclear>`_ for nuclei.


Hard constraints
````````````````

This section is about adjusting the parameters so we have hypotheses that include the correct solution.

Please refer to the :doc:`Configuration docs <configuration>` as we refer to different parameters.

1. The expected cell size should be between ``segmentation_config.min_area`` and ``segmentation_config.max_area``.
Having a tight range assists in finding a good segmentation and significantly reduces the computation.
Our rule of thumb is to set the ``min_area`` to half the size of the expected cell or the smallest cell, *disregarding outliers*.
And the ``max_area`` to 1.25~1.5 the size of the largest cell, this is less problematic than the ``min_area``.

2. ``linking_config.max_distance`` should be set to the maximum distance a cell can move between frames.
We recommend setting some tolerance, for example, 1.5 times the expected movement.

Tracking tuning
```````````````

Once you have gone through the previous steps, you should have a working pipeline and now we can focus on the results and what can be done in each scenario.

1. My cells are oversegmented (excessive splitting of cells):
    - Increase the ``segmentation_config.min_area`` to merge smaller cells;
    - Increase the ``segmentation_config.max_area`` to avoid splitting larger cells;
    - If you have clear boundaries and the oversegmentation are around weak boundaries, you can increase the ``segmentation_config.min_frontier`` to merge them (steps of 0.05 recommended).
    - If you're using ``labels`` as input or to create my contours you can also try to increase the ``sigma`` parameter to create a better surface to segmentation by avoiding flat regions (full of zeros or ones).

2. My cells are undersegmented (cells are fused):
    - Decrease the ``segmentation_config.min_area`` to enable segmenting smaller cells;
    - Decrease the ``segmentation_config.max_area`` to remove larger segments that are likely to be fused cells;
    - Decrease the ``segmentation_config.min_frontier`` to avoid merging cells that have weak boundaries;
    - **EXPERIMENTAL**: Set ``segmentation_config.max_noise`` to a value greater than 0, to create more diverse hierarchies, the scale of this value should be proportional to the ``contours`` value, for example, if the ``contours`` is in the range of 0-1, the ``max_noise`` around 0-0.05 should be enough. Play with it. **NOTE**: the solve step will take longer because of the increased number of hypotheses.

3. I have missing segments that are present on the ``labels`` or ``foreground``:
    - Check if these cells are above the ``segmentation_config.threshold`` value, if not, decrease it;
    - Check if ``linking_config.max_distance`` is too low and increase it, when cells don't have connections they are unlikely to be included in the solutions;
    - Your ``tracking_config.appear_weight``, ``tracking_config.disappear_weight`` & ``tracking_config.division_weight`` penalization weights are too high (too negative), try bringing them closer to 0.0. **TIP**: We recommend adjusting ``disappear_weight`` weight first, because when tuning ``appear_weight`` you should balance out ``division_weight`` so appearing cells don't become fake divisions. A rule of thumb is to keep ``division_weight`` equal or higher (more negative) than ``appear_weight``.

4. I'm not detecting enough dividing cells:
    - Bring ``tracking_config.division_weight`` to a value closer to 0.
    - Depending on your time resolution and your cell type, it might be the case where dividing cells move further apart, in this case, you should tune the ``linking_config.max_distance`` accordingly.

5. I'm detecting too many dividing cells:
    - Make ``tracking_config.division_weight`` more negative.

6. My tracks are short and not continuous enough:
    - This is tricky, once you have tried the previous steps, you can try making the ``tracking_config.{appear, division, disappear}_weight`` more negative, but this will remove low-quality tracks.
    - Another option is to use ``ultrack.tracks.close_tracks_gaps`` to post process the tracks.

7. I have many incorrect tracks connecting distant cells:
    - Decrease the ``linking_config.max_distance`` to avoid connecting distant cells. If that can't be done because you will lose correct connections, then you should set ``linking_config.distance_weight`` to a value closer higher than 0, usually in very small steps (0.01).
