ARG PYTORCH_VERSION=2.5.1-cuda11.8-cudnn9
ARG ULTRACK_VERSION=0.6.1

FROM pytorch/pytorch:${PYTORCH_VERSION}-devel

ARG PYTORCH_VERSION
ARG ULTRACK_VERSION

WORKDIR /app

RUN apt update &&  \
    apt install -y --no-install-recommends && \
    apt install -y libgl1-mesa-dev grep && \
    rm -rf /var/lib/apt/lists/*

RUN export CU_VER=$(echo "${PYTORCH_VERSION}" | grep -oP "cuda\K[0-9]+") && \
    pip install --no-cache-dir --root-user-action ignore uv && \
    conda install -y -c conda-forge pyqt && \
    uv pip install --no-cache --system "cucim-cu${CU_VER}>=24.10.0" gurobipy higra && \
    pip install --no-cache --root-user-action ignore "ultrack==${ULTRACK_VERSION}" && \
    conda clean -ay

ENTRYPOINT ["/usr/bin/bash"]