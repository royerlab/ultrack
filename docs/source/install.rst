Installation
============

The simplest way to install ``ultrack`` is with pip:

.. code-block:: bash

    pip install ultrack

We recommend installing into a virtual environment to avoid package conflicts.
With Python's built-in ``venv`` on Linux / macOS:

.. code-block:: bash

    python -m venv ultrack-env
    source ultrack-env/bin/activate
    pip install ultrack

On Windows:

.. code-block:: bat

    python -m venv ultrack-env
    ultrack-env\Scripts\activate
    pip install ultrack

Or with `conda <https://docs.conda.io/projects/conda/en/latest/>`_:

.. code-block:: bash

    conda create -n ultrack python=3.11
    conda activate ultrack
    pip install ultrack

You can verify the installation by running:

.. code-block:: bash

    ultrack --help


GPU acceleration
----------------

Ultrack uses the GPU for image processing operations.
Install the additional packages for GPU acceleration (Linux and Windows only):

.. code-block:: bash

    # Install PyTorch with CUDA — pick the right version at https://pytorch.org/get-started/locally/
    pip install torch --index-url https://download.pytorch.org/whl/cu124
    pip install cupy-cuda12x
    # Linux only
    pip install cucim-cu12

See the `PyTorch website <https://pytorch.org/get-started/locally/>`_ for the full list of CUDA builds.

.. _gurobi_install:

Gurobi setup
------------

Gurobi is a commercial optimization solver used in the tracking module of ``ultrack``.
While not required, it is recommended for best performance.

``gurobipy`` is installed automatically with ``ultrack`` — no extra install step is needed.
To use the solver, you must obtain and activate a license (free for academics).

Obtain and activate an academic license
```````````````````````````````````````

**Obtaining a license:** register for an account using your academic email at `Gurobi's website <https://portal.gurobi.com/iam/login>`_.
Navigate to the Gurobi's `named academic license page <https://www.gurobi.com/features/academic-named-user-license/>`_, and follow the instructions to get your academic license key.

**Activating license:** run:

.. code-block:: bash

    grbgetkey YOUR_LICENSE_KEY

Replace ``YOUR_LICENSE_KEY`` with the key you received. Follow the prompts to complete activation.

Test the installation
`````````````````````

Verify Gurobi's installation by running:

.. code-block:: bash

    ultrack check_gurobi

Troubleshooting
```````````````

Depending on the operating system, the Gurobi library might be missing.
Download it from `Gurobi's website <https://www.gurobi.com/downloads/gurobi-software>`_.

If you're still having trouble, reach out to us or use the Docker image — see :doc:`Docker instructions <docker/README>`.

.. _pixi_install:

Ultrack environment with pixi
-----------------------------

For contributors and advanced users, we provide a `pixi <https://pixi.sh/>`_ configuration
that sets up a full development environment in a single command.

Install ``pixi`` following the instructions for your operating system:

For Linux and macOS:

.. code-block:: bash

    curl -fsSL https://pixi.sh/install.sh | bash

For Windows (PowerShell):

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
