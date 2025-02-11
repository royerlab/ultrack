FAQ
---

**Q: What is each configuration parameters for?**
    A: See the :doc:`configuration page <configuration>`.

**Q: What to do when Qt platform plugin cannot be initialized?**
    A: The solution to try is to install ``pyqt`` using ``conda`` from the ``-c conda-forge`` channel.

**Q: Why my python script gets stuck when using ultrack?**
    A: You need to wrap your code in a ``if __name__ == '__main__':`` block to avoid the ``multiprocessing`` module to run the same code in each process.
    For example:

    .. code-block:: python

        import ultrack

        def main():
            # Your code here
            ...

        if __name__ == '__main__':
            main()

**Q: My results show strange segments with perfect lines boundaries. What is happening and how can I fix it?**
    A: This is a hierarchical watershed artifact. Regions with "flat" intensities create arbitrary partitions that are, most of the time, a straight line.

    You have three options to fix this:

    - increase ``min_area`` parameter, so these regions get removed. However, if you have objects with varying areas, this might be challenging and lead to undersegmentation.
    - increase ``min_frontier``; this is my preferred option when you have a high accuracy network as plants.
        This merges regions whose average intensity between them is below min_frontier.
        In this case, assuming your boundary is between 0 and 1, ``min_frontier=0.1`` or even ``0.05`` should work.
        Be careful to not increase this too much because it could merge regions where cells are shared by a "weak" boundary.
    - another option is to blur the boundary map so you avoid creating regions with "flat" intensities.
        This follows the same reasoning for using EDT to run watersheds.
        This works better for convex objects. And remember to renormalize the intensity values if using this with ``min_frontier``.

**Q: Ultrack is not working with my data. What can I do?**
    A: See the :doc:`tracking optimizing page <optimizing>`.

**Q: My data is isotropic. How can I take that into account?**
    A: Provide the the Z, Y, X scaling factors in the ``scale`` parameter of ``track`` or ``link`` functions.

**Q: How can I use Ultrack distributed over a cluster?**
    A: `Jacky Ko <https://github.com/jackyko1991>`_ shared his setup and his documentation `here <https://github.com/jackyko1991/Ultrack-Cluster>`_.
