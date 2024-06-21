Ultrack's Usage Examples
========================

Here we provide some examples of how to use Ultrack for cell tracking.

Some examples are provided as Jupyter notebooks with additional documentation, but we do not recommend using Jupyter notebooks for your day-to-day analysis.

Other examples as Python scripts can be found in `here <https://github.com/royerlab/ultrack/tree/main/examples>`_.

Additional packages might be required. Therefore, conda environment files are provided, which can be installed using:

.. code-block:: bash

    conda env create -f <environment-file.yml>
    conda activate <your new env>
    pip install git+https://github.com/royerlab/ultrack

The existing examples are:

- `multi_color_ensemble <./multi_color_ensemble>`_ : Multi-colored cytoplasm cell tracking using Cellpose and Watershed segmentation ensemble. Data provided by `The Lammerding Lab <https://lammerding.wicmb.cornell.edu/>`_.
- `flow_field_3d <./flow_field_3d>`_ : Tracking demo on a cartographic projection of Tribolium Castaneum embryo from the `cell-tracking challenge <http://celltrackingchallenge.net/3d-datasets/>`_, using a flow field estimation to assist tracking of motile cells.
- `stardist_2d <./stardist_2d>`_ : Tracking demo on HeLa GPF nuclei from the `cell-tracking challenge <http://celltrackingchallenge.net/2d-datasets/>`_ using Stardist 2D fluorescence images pre-trained model.
- `zebrahub <./zebrahub/>`_ : Tracking demo on zebrafish tail data from `zebrahub <https://zebrahub.ds.czbiohub.org/>`_ acquired with `DaXi <https://www.nature.com/articles/s41592-022-01417-2>`_ using Ultrack's image processing helper functions.
- `neuromast_plantseg <./neuromast_plantseg/>`_ : Tracking demo membrane-labeled zebrafish neuromast from `Jacobo Group of CZ Biohub <https://www.czbiohub.org/jacobo/>`_ using `PlantSeg's <https://github.com/hci-unihd/plant-seg>`_ membrane detection model.
- `micro_sam <./micro_sam/>`_ : Tracking demo with `MicroSAM <https://github.com/computational-cell-analytics/micro-sam>`_ instance segmentation package.

Development Notes
^^^^^^^^^^^^^^^^^

To run all the examples and update the notebooks in headless mode, run:

.. code-block:: bash

    bash refresh_examples.sh
