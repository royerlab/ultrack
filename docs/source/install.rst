Installation
============

The recommended way to install the package is to use the conda (or mamba) package manager.

We also provide a `pixi <https://pixi-docs.com/>`_ configuration for the package, which allows you to setup an environment with all the dependencies in a single command. See the section :ref:`pixi_install` for more details.

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
    pip install "git+https://github.com/rapidsai/cucim.git#egg=cucim&subdirectory=python/cucim"

See the `PyTorch website <https://pytorch.org/get-started/locally/>`_ for more information on how to install PyTorch with GPU support.

.. _gurobi_install:

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

.. _pixi_install:

Ultrack environment with pixi
-----------------------------

This is an alternative to the conda installation for environment management, using the `pixi <https://pixi-docs.com/>`_ package manager.

First, install ``pixi`` following the instructions for your operating system:

For Linux and OSX:

.. code-block:: bash

    curl -fsSL https://pixi.sh/install.sh | bash

For Windows, using PowerShell:

.. code-block:: powershell

    powershell -ExecutionPolicy ByPass -c "irm -useb https://pixi.sh/install.ps1 | iex"

Clone the repository and navigate to it:

.. code-block:: bash

    git clone https://github.com/ultrack/ultrack.git
    cd ultrack

Install dependencies and create the environment:

.. code-block:: bash

    pixi install

There are two ways to work with the pixi environment:

1. Activate the environment (equivalent to ``conda activate``):

.. code-block:: bash

    pixi shell

2. Run a single command within the environment:

.. code-block:: bash

    pixi run python your_script.py

The environment will automatically detect and use CUDA if it's available on your system.
