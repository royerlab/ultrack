
# ULTRACK

Multiple hypothesis cell tracking using ultrametric-contour maps.

## Installation

Preliminary installation instructions while this repository is private and the package is not in pypi.

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

Install the package from github, you must have setup your [SSH keys](https://docs.github.com/en/authentication/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) and have access to the [ultrack repository](https://github.com/royerlab/ultrack) since the repository is private.

```bash
pip install git+ssh://git@github.com/royerlab/ultrack.git
```

## Usage

Here you can find a usage example: https://github.com/royerlab/ultrack/blob/main/examples/stardist_2d/2d_tracking.ipynb
