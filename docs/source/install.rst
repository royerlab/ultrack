Installation
============

The easiest way to install the package is to use the conda (or mamba) package manager.
If you do not have conda installed, we recommend to install mamba first, which is a faster alternative to conda.
You can find mamba installation instructions `here <https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html>`_.

Once you have conda (mamba) installed, you should create an environment for ``ultrack`` as follows:

.. code-block:: bash

    conda create -n ultrack python=3.11 higra gurobi pytorch pyqt -c pytorch -c gurobi -c conda-forge

Then, you can activate the environment and install ``ultrack``:

.. code-block:: bash

    conda activate ultrack
    pip install ultrack

You can check if the installation was successful by running:

.. code-block:: bash

    ultrack --help


GPU acceleration
----------------

Ultrack makes use of GPU for image processing operations.
You can install the additional packages required for GPU acceleration by running (Linux and Windows only):

.. code-block:: bash

    conda install pytorch-cuda -c pytorch -c nvidia
    conda install cupy -c conda-forge
    # linux only
    conda install cucim -c rapidsai
    # for windows, you can install cucim using pip
    pip install git+https://github.com/rapidsai/cucim.git#egg=cucim&subdirectory=python/cucim"

See the `PyTorch website <https://pytorch.org/get-started/locally/>`_ for more information on how to install PyTorch with GPU support.

Gurobi setup
------------

Gurobi is a commercial optimization solver that is used in the tracking module of ``ultrack``.
While it is not a requirement, it is recommended to install it for the best performance.

To use it, you need to obtain a license (free for academics) and activate it.

Install gurobi using conda
``````````````````````````

You can skip this step if you have already installed Gurobi.

In your existing Conda environment, install Gurobi with the following command:

.. code-block:: bash

    conda install -c gurobi gurobi

Obtain and activate an academic license
```````````````````````````````````````

**Obtaining a license:** register for an account using your academic email at `Gurobi's website <https://portal.gurobi.com/iam/login>`_.
Navigate to the Gurobi's `named academic license page <https://www.gurobi.com/features/academic-named-user-license/>`_, and follow the instructions to get your academic license key.

**Activating license:** In your Conda environment, run:

.. code-block:: bash

    grbgetkey YOUR_LICENSE_KEY

Replace YOUR_LICENSE_KEY with the key you received. Follow the prompts to complete activation.

Test the installation
`````````````````````

Verify Gurobi's installation by running:

.. code-block:: bash

    ultrack check_gurobi

Troubleshooting
```````````````

Depending on the operating system, the gurobi library might be missing and you need to install it from `here <https://www.gurobi.com/downloads/gurobi-software>`_.

If you're still having trouble, with the installation we recommend reaching out to us or using the docker image, see  :doc:`Docker instructions <docker/README>`.
