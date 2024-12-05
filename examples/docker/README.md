# Ultrack Docker Image

This example shows how to use the Ultrack Docker image. The Docker image is a pre-configured environment that includes 
the Ultrack software and all its dependencies. The images are available on Docker Hub at [TBD](https://hub.docker.com/).
Also, you can build the image from scratch (see the [Building the Docker Image](#optional-building-the-docker-image) section).

In the following sections, we will discuss:

- [Running the Docker Image](#running-the-docker-image) for basic introduction to the Ultrack software inside the Docker 
    container.
- [Gurobi support](#gurobi-support) for using Gurobi inside the Ultrack Docker image.
- [GPU support](#gpu-support) for using GPU acceleration inside the Ultrack Docker image.
- [Running the Docker Image with Jupyter Notebook](#running-the-docker-image-with-jupyter-notebook) for running a Jupyter
    Notebook server inside the Docker container.
- [Building the Docker Image](#optional-building-the-docker-image) for building the Ultrack Docker image from scratch.

## Running the Docker Image

To run the Ultrack Docker image, you can use the following command:

```bash
docker run -it --rm -v /path/to/your/data:/data ultrack/0.6.1-cpu  # replace 6.0.1-cpu with the desired version and 
                                                                   # variant (e.g., 6.0.1-cpu, 6.0.1-cuda11.8)
```

This command will start the Ultrack Docker image and mount the `/path/to/your/data` directory to the `/data` directory
inside the container. You can replace `/path/to/your/data` with the path to your data directory. Then, you can use the
[Ultrack CLI](https://royerlab.github.io/ultrack/cli.html) for your analysis.

## Gurobi support

Gurobi is a commercial optimization solver that is used in Ultrack for solving the optimization problems. To use Gurobi
inside the Ultrack Docker image, you need to have a valid 
[Web License Service (WLS)](https://www.gurobi.com/features/web-license-service/) license. Once you have the license file
(`gurobi.lic`), you can mount it to the `/opt/gurobi/gurobi.lic` directory inside the container. For example, you can
improve the previous command to include the Gurobi license file as follows:

```bash
docker run -it --rm -v /path/to/your/data:/data -v /path/to/your/gurobi.lic:/opt/gurobi/gurobi.lic ultrack/0.6.1-cpu
```

And now check the Gurobi support by running the following command:

```bash
ultrack check_gurobi
```

which should display the Gurobi license information if the license file is valid.

## GPU support

The Ultrack Docker image also supports GPU acceleration using CUDA. To use the GPU support, you need to have a compatible
NVIDIA GPU and the NVIDIA driver installed on your host machine. Then, you need to include the `--gpus all` flag in the
`docker run` command. For example, you can run the Ultrack Docker image with GPU support as follows:

```bash
docker run -it --rm -v /path/to/your/data:/data --gpus all ultrack/0.6.1-cuda11.8  # replace 6.0.1-cuda11.8 with the 
                                                                                   # desired version and variant
```

## Running the Docker Image with Jupyter Notebook

You can also run the Ultrack Docker image with a Jupyter Notebook server. To do this, you need to forward the Jupyter
Notebook port (default is 8888) to the host machine by `-p 8888:8888`. Make sure that the port is not already in use on
the host machine. Otherwise, you can use a different port number (e.g., `-p 8889:8888`). Then, proceed with the 
following command:

```bash
docker run -it --rm -v /path/to/your/data:/data -p 8888:8888 ultrack/0.6.1-cpu
```

After running the command, you will need to install the Jupyter Notebook server by running the following command:

```bash
uv pip install --no-cache --system jupyterlab  # or simply `pip install jupyterlab`
```
Then, you can start the Jupyter Notebook server by running the following command:

```bash
jupyter lab --ip=0.0.0.0 --allow-root  # ip=0.0.0.0 allows access from any IP address (e.g. outside the container)
                                       # allow-root allows running Jupyter Notebook as root user, which is the default user inside the container
```

After running the command, you will see a URL that you can open in your browser to access the Jupyter Notebook server.
Then, you can use the default [Ultrack API](https://royerlab.github.io/ultrack/api.html) to analyze your data.

## (Optional) Building the Docker Image 

If you want to build the Ultrack Docker image from scratch, you can use the following command:

```bash
python3 build_containers.py 
```

This command requires an argument that specifies the desired version and variant of the Ultrack Docker image. If not 
provided, the script will list all available versions and variants. For example, you can build the Ultrack Docker image
with basic CPU support by running the following command:

```bash
python3 build_containers.py cpu
```

And now you can use the built image as described in the [Running the Docker Image](#running-the-docker-image) section.