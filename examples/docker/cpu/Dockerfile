ARG PYTHON_VERSION=3.11
ARG ULTRACK_VERSION=0.6.1

FROM ubuntu:22.04

ARG PYTHON_VERSION
ARG ULTRACK_VERSION

WORKDIR /app

RUN apt update &&  \
    apt install -y --no-install-recommends && \
    apt install -y libgl1-mesa-dev grep curl && \
    rm -rf /var/lib/apt/lists/*

RUN curl -fsSL -v -o ~/miniconda.sh -O  "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh" && \
    chmod +x ~/miniconda.sh && \
    bash ~/miniconda.sh -b -p /opt/conda && \
    rm ~/miniconda.sh && \
    /opt/conda/bin/conda install -y python=${PYTHON_VERSION}

ENV PATH=/opt/conda/bin:$PATH

RUN conda install -y -c conda-forge pyqt uv && \
    uv pip install --no-cache --system gurobipy higra && \
    pip install --no-cache --root-user-action ignore "ultrack==${ULTRACK_VERSION}" && \
    conda clean -ay

ENTRYPOINT ["/usr/bin/bash"]