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

The official documentation is available [here](https://royerlab.github.io/ultrack/).

These additional developer documentation are available:

- Parameter [configuration schema](ultrack/config/README.md).
- Intermediate segmentation and tracking SQL database are [here](ultrack/core/README.md).

## Gurobi setup

Installing gurobi and setting up an academic license.

### Install Gurobi using Conda

In your existing Conda environment, install Gurobi with the following command:

```bash
conda install -c gurobi gurobi
```

### Obtain and Activate an Academic License

**Obtain License:** register for an account using your academic email at [Gurobi's website](https://portal.gurobi.com/iam/login/). Navigate to the Gurobi's [named academic license page](https://www.gurobi.com/features/academic-named-user-license/), and follow instructions to get your academic license key.

**Activate License:** In your Conda environment, run:

```bash
grbgetkey YOUR_LICENSE_KEY
```

Replace YOUR_LICENSE_KEY with the key you received. Follow the prompts to complete activation.

### Test the Installation

Verify Gurobi's installation by running:

```bash
ultrack check_gurobi
```

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
