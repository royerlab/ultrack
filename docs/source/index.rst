.. ultrack
   =======

ultrack
=======

``ultrack`` is general purpose 2D/3D cell tracking software.

It can track from segmentation masks or from raw images directly, specially fluorescence microscopy images.

Four interfaces are provided, depending on your needs:

- napari plugin
- FIJI plugin
- Python API
- Command line interface for batch processing, including distributed computing.

See below for additional details in each interface.


Moreover, it was originally developed to terabyte-scale zebrafish embryo images where we had few 3D annotations.
Hence, a few key features of ``ultrack`` are:

- Out-of-memory storage of intermediate results. You should not run out of memory even for large datasets. We have tracked a 3TB dataset on a laptop with 64GB of RAM.
- It does not commit to a single segmentation. Instead it considers multiple segmentations per cell and it picks the best segmentation for each cell at each time point while tracking.

.. raw:: html

   <video width="480" autoplay muted>
     <source src="https://github.com/royerlab/ultrack/assets/21022743/10aace9c-0e0e-4310-a103-f846683cfc77" type="video/mp4">
     Your browser does not support the video tag. Imagine cells being tracked in 3D image.
   </video>

Zebrafish imaged using `DaXi <https://www.nature.com/articles/s41592-022-01417-2>`_ whole embryo tracking.

---------------

.. include:: quickstart.rst

---------------

.. include:: install.rst

---------------

.. include:: napari.rst

---------------

.. include:: fiji.rst

---------------

.. include:: getting_started.rst

---------------

.. include:: docker/README.rst

---------------

.. include:: examples.rst
   :end-line: 6

---------------

.. include:: optimizing.rst

---------------

.. include:: configuration.rst

---------------

.. include:: faq.rst

---------------

.. include:: theory.rst

---------------


Citing
------

If you use ``ultrack`` in your research, please cite the following papers, `the algorithm <https://arxiv.org/pdf/2308.04526>`_ and `the biological applications and software <https://www.biorxiv.org/content/10.1101/2024.09.02.610652>`_.

.. code-block:: bibtex

   @article{bragantini2023ucmtracking,
      title={Large-Scale Multi-Hypotheses Cell Tracking Using Ultrametric Contours Maps},
      author={Jordão Bragantini and Merlin Lange and Loïc Royer},
      year={2023},
      eprint={2308.04526},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
   }

   @article{bragantini2024ultrack,
      title={Ultrack: pushing the limits of cell tracking across biological scales},
      author={Bragantini, Jordao and Theodoro, Ilan and Zhao, Xiang and Huijben, Teun APM and Hirata-Miyasaki, Eduardo and VijayKumar, Shruthi and Balasubramanian, Akilandeswari and Lao, Tiger and Agrawal, Richa and Xiao, Sheng and others},
      journal={bioRxiv},
      pages={2024--09},
      year={2024},
      publisher={Cold Spring Harbor Laboratory}
   }

And the respective auxiliary methods (e.g. Cellpose, napari, etc) depending on your usage.


Additional resources
--------------------

.. [1] `Jacky Ko <https://github.com/jackyko1991>`_'s Ultrack cluster setup and documentation: https://github.com/jackyko1991/Ultrack-Cluster

Documentation contents
``````````````````````

.. toctree::
   :maxdepth: 1
   :caption: Basics:

   quickstart
   install
   napari
   fiji
   getting_started
   examples
   optimizing
   configuration
   docker/README
   faq
   theory

.. toctree::
   :caption: Reference:

   api
   cli
   rest_api
