
# ULTRACK

Cell tracking and segmentation software.

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
pip install git+https://github.com/royerlab/ultrack.git
```

## Usage

Usage examples can be found [here](examples), including their environment files and their installation instructions.

## Documentation

The `ultrack` library relies on a configuration schema, its description is [here](ultrack/config/README.md).

The segmentation and tracking data are stored in a SQL database, described [here](ultrack/core/README.md).

Helper functions to export to the cell tracking challenge and napari formats are available, [reference](ultrack/core/export).
