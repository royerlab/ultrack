Ultrack Docker Image
====================

This example shows how to use the Ultrack Docker image. The Docker image is a pre-configured environment that includes
the Ultrack software and all its dependencies. The images are available on Docker Hub at
`royerlab/ultrack repository <https://hub.docker.com/r/royerlab/ultrack/tags>`_. You can also build the image from scratch (see the
:ref:`running-the-docker-image` section).

In the following sections, we will discuss:

- :ref:`running-the-docker-image` for a basic introduction to the Ultrack software inside the
  Docker container.
- :ref:`gurobi-support` for using Gurobi inside the Ultrack Docker image.
- :ref:`gpu-support` for using GPU acceleration inside the Ultrack Docker image.
- :ref:`running-the-docker-image-with-jupyter-notebook` for running a
  Jupyter Notebook server inside the Docker container.
- :ref:`optional-building-the-docker-image` for building the Ultrack Docker image from scratch.

.. _running-the-docker-image:
Running the Docker Image
------------------------

To run the Ultrack Docker image, you can use the following command:

.. code-block:: bash

    docker run -it --rm -v /path/to/your/data:/data royerlab/ultrack:0.6.1-cpu  # replace 6.0.1-cpu with the desired version and
                                                                                # variant (e.g., 6.0.1-cpu, 6.0.1-cuda11.8)

This command will start the Ultrack Docker image and mount the `/path/to/your/data` directory to the `/data` directory
inside the container. Replace `/path/to/your/data` with the path to your data directory. Then, you can use the
`Ultrack CLI <https://royerlab.github.io/ultrack/cli.html>`_ for your analysis.

.. _gurobi-support:
Gurobi support
--------------

Gurobi is a commercial optimization solver used in Ultrack for solving optimization problems. To use Gurobi inside the
Ultrack Docker image, you need a valid
`Web License Service (WLS) <https://www.gurobi.com/features/web-license-service/>`_ license. Once you have the license
file (`gurobi.lic`), mount it to the `/opt/gurobi/gurobi.lic` directory inside the container. For example:

.. code-block:: bash

    docker run -it --rm -v /path/to/your/data:/data \
           -v /path/to/your/gurobi.lic:/opt/gurobi/gurobi.lic \
           royerlab/ultrack:0.6.1-cpu

Check the Gurobi support by running:

.. code-block:: bash

    ultrack check_gurobi

This should display the Gurobi license information if the license file is valid.

.. _gpu-support:
GPU support
-----------

The Ultrack Docker image supports GPU acceleration using CUDA. To use GPU support, you need a compatible NVIDIA GPU and
the NVIDIA driver installed on your host machine. Include the `--gpus all` flag in the `docker run` command. For example:

.. code-block:: bash

    docker run -it --rm -v /path/to/your/data:/data --gpus all \
           royerlab/ultrack:0.6.1-cuda11.8  # replace 6.0.1-cuda11.8 with the
                                            # desired version and variant

.. _running-the-docker-image-with-jupyter-notebook:
Running the Docker Image with Jupyter Notebook
----------------------------------------------

You can run the Ultrack Docker image with a Jupyter Notebook server by forwarding the default Jupyter port (8888) to the
host machine using `-p 8888:8888`. If the port is in use, use a different port (e.g., `-p 8889:8888`):

.. code-block:: bash

    docker run -it --rm -v /path/to/your/data:/data -p 8888:8888 ultrack/0.6.1-cpu

After starting the container, install the Jupyter Notebook server:

.. code-block:: bash

    uv pip install --no-cache --system jupyterlab  # or simply `pip install jupyterlab`

Start the Jupyter Notebook server:

.. code-block:: bash

    jupyter lab --ip=0.0.0.0 --allow-root  # ip=0.0.0.0 allows access from any IP address
                                           # allow-root allows running Jupyter as root (default user in container)

You will see a URL to access the Jupyter Notebook server in your browser. Use the
`Ultrack API <https://royerlab.github.io/ultrack/api.html>`_ to analyze your data.

.. _optional-building-the-docker-image:
(Optional) Building the Docker Image
------------------------------------

To build the Ultrack Docker image from scratch, use the following command:

.. code-block:: bash

    python3 docker/build_containers.py

This command requires an argument specifying the desired version and variant of the Ultrack Docker image. If not
provided, the script lists all available versions and variants. For example, to build an image with basic CPU support:

.. code-block:: bash

    python3 docker/build_containers.py cpu

You can then use the built image as described in the
:ref:`running-the-docker-image` section.
