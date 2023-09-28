# ULTRACK

![tests](https://github.com/royerlab/ultrack/actions/workflows/test_pull_request.yml/badge.svg)
[![codecov](https://codecov.io/gh/royerlab/ultrack/branch/main/graph/badge.svg?token=9FFo4zNtYP)](https://codecov.io/gh/royerlab/ultrack)
[![PyPI version](https://badge.fury.io/py/ultrack.svg)](https://badge.fury.io/py/ultrack)
[![Downloads](https://pepy.tech/badge/ultrack)](https://pepy.tech/project/ultrack)
[![Downloads](https://pepy.tech/badge/ultrack/month)](https://pepy.tech/project/ultrack)
[![Python version](https://img.shields.io/pypi/pyversions/ultrack)](https://pypistats.org/packages/ultrack)
[![Licence: BSD-3](https://img.shields.io/github/license/royerlab/ultrack)](https://github.com/royerlab/ultrack/blob/main/LICENSE)
[![Contributors](https://img.shields.io/github/contributors-anon/royerlab/ultrack)](https://github.com/royerlab/ultrack/graphs/contributors)
[![GitHub stars](https://img.shields.io/github/stars/royerlab/ultrack?style=social)](https://github.com/royerlab/ultrack/)
[![GitHub forks](https://img.shields.io/github/forks/royerlab/ultrack?style=social)](https://git:hub.com/royerlab/ultrack/)

Large-scale cell tracking under segmentation uncertainty.

https://github.com/royerlab/ultrack/assets/21022743/10aace9c-0e0e-4310-a103-f846683cfc77

Zebrafish imaged using [DaXi](https://www.nature.com/articles/s41592-022-01417-2) whole embryo tracking.

## Installation

Install [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html) to avoid conflicts between different packages.

Create a conda environment.

```bash
conda create --name tracking python=3.10
```

And activate it.

**ATTENTION**: every time you need to run this software you'll have to activate this environment

```bash
conda activate tracking
```

Install the package from github:

```bash
pip install git+https://github.com/royerlab/ultrack
```

## Usage

Usage examples can be found [here](examples), including their environment files and their installation instructions.

## Documentation

The `ultrack` library relies on a configuration schema, its description is [here](ultrack/config/README.md).

The segmentation and tracking data are stored in an SQL database, described [here](ultrack/core/README.md).

Helper functions to export to the cell tracking challenge and napari formats are available [here](ultrack/core/export).

## Citing

```
@misc{bragantini2023ultrack,
      title={Large-Scale Multi-Hypotheses Cell Tracking Using Ultrametric Contours Maps},
      author={Jordão Bragantini and Merlin Lange and Loïc Royer},
      year={2023},
      eprint={2308.04526},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```
