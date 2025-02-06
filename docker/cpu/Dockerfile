ARG PYTHON_VERSION=3.11
ARG ULTRACK_VERSION=0.6.1

FROM ubuntu:22.04

ARG PYTHON_VERSION
ARG ULTRACK_VERSION

WORKDIR /app

RUN apt update &&  \
    apt install -y --no-install-recommends && \
    apt install -y libgl1-mesa-dev grep curl xvfb tini && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -v -o ~/miniconda.sh -O  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" && \
    chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION}

ENV PATH=/opt/conda/bin:$PATH

RUN conda install -y -c conda-forge pyqt uv && \
    uv pip install --no-cache --system torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu && \
    uv pip install --no-cache --system higra && \
    pip install --no-cache --root-user-action ignore "ultrack==${ULTRACK_VERSION}" && \
    # avoid gurobi package from pypi since version 12.0.0 is broken
    pip uninstall --no-cache --root-user-action ignore gurobipy -y && \
    conda install -y -c gurobi gurobi && \
    conda clean -ay

# tini is needed to run xvfb-run
# xvfb-run is needed to run GUI applications in headless mode (e.g. napari-reader)
ENTRYPOINT ["tini", "--", "xvfb-run"]
CMD ["/usr/bin/bash"]
