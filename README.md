
<div align="left">
<picture>
  <!-- loads when the visitor is in dark‑mode -->
  <source media="(prefers-color-scheme: dark)" srcset="logo/ultrack_dark_bkg.svg" />

  <!-- loads when the visitor is in light‑mode -->
  <source media="(prefers-color-scheme: light)" srcset="logo/ultrack_no_bkg.svg" />

  <!-- fallback if the browser doesn’t understand <picture> -->
  <img alt="Ultrack Logo" src="logo/ultrack_no_bkg.svg" style="width:400px;"  />
</picture>
</div>

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

## Overview

Ultrack is a versatile and scalable cell tracking method designed to address the challenges of tracking cells across 2D, 3D, and multichannel timelapse recordings, especially in complex and crowded tissues where segmentation is often ambiguous. By evaluating multiple candidate segmentations and employing temporal consistency, Ultrack ensures robust performance under segmentation uncertainty. Ultrack's methodology is explained [here](https://arxiv.org/pdf/2308.04526).

https://github.com/royerlab/ultrack/assets/21022743/10aace9c-0e0e-4310-a103-f846683cfc77

Zebrafish imaged using [DaXi](https://www.nature.com/articles/s41592-022-01417-2) whole embryo tracking.

## Features

- **Versatile Cell Tracking:** Supports 2D, 3D, and multichannel datasets.
- **Robust Under Segmentation Uncertainty:** Evaluates multiple candidate segmentations.
- **High Performance:** Scales from small in vitro datasets to terabyte-scale developmental time-lapses.
- **Integration:** Compatible with FiJi, napari, and high-performance clusters via SLURM.

## Installation

Install or update [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html).

To avoid conflicts between different packages, we recommend using conda to create an isolated environment:

```bash
conda create -n ultrack python=3.11 higra gurobi pytorch pyqt -c pytorch -c gurobi -c conda-forge
conda activate ultrack
pip install ultrack
```

The installation should take a few minutes, depending on your internet speed and conda.

NOTE: `gurobi` and `-c gurobi` are optional but recommended; they can be installed later, as shown below.

Optionally, we provide multiple Docker images. For instructions, see the [docker folder](https://github.com/royerlab/ultrack/tree/main/docker).

## Usage

**ATTENTION**: every time you need to run this software, you'll have to activate this environment

```bash
conda activate ultrack
```

Here is a basic example to get you started:

```python
import napari
from ultrack import MainConfig, Tracker

# __main__ is recommended to avoid multi-processing errors
if __name__ == "__main__":
      # Load your data
      foreground = ...
      contours = ...

      # Create config
      config = MainConfig()

      # Run tracking
      tracker = Tracker(config)
      tracker.track(foreground=foreground, edges=contours)

      # Visualize results in napari
      tracks, graph = tracker.to_tracks_layer()
      napari.view_tracks(tracks[["track_id", "t", "z", "y", "x"]], graph=graph)
      napari.run()
```

More usage examples can be found [here](examples), including their environment files and installation instructions.

## Documentation

Comprehensive documentation is available [here](https://royerlab.github.io/ultrack/).

These additional developer documentation are available:

- Parameter [configuration schema](docs/source/configuration.rst).
- Intermediate segmentation and tracking SQL database are [here](ultrack/core/README.md).

## Gurobi Setup

### Install Gurobi using Conda

In your existing Conda environment, install Gurobi with the following command:
```bash
conda install -c gurobi gurobi
```

### Obtain and Activate an Academic License

1. Register at [Gurobi's website](https://portal.gurobi.com/iam/login/) with your academic email.
2. Navigate to the Gurobi's [named academic license page](https://www.gurobi.com/features/academic-named-user-license/)
3. Follow the instructions to get your license key.
4. Activate your license, In your Conda environment, run:

```bash
grbgetkey YOUR_LICENSE_KEY
```

5. Replace YOUR_LICENSE_KEY with the key you received. Follow the prompts to complete activation.

### Verify Installation

Verify Gurobi's installation by running:
```bash
ultrack check_gurobi
```

Depending on the operating system, the gurobi library might be missing and you need to install it from [here](https://www.gurobi.com/downloads/gurobi-software).

## Who is using Ultrack?

You can find a list of projects and papers that have used ultrack on [this page](https://royerlab.github.io/ultrack/appearances.html).

## Contributing

We welcome contributions from the community! To get started, please read our [contributing guidelines](CONTRIBUTING.md). Then, report issues and submit pull requests on GitHub.

## License

This project is licensed under the BSD-3 License - see the [LICENSE](LICENSE) file for details.

## Citing

If you use `ultrack` in your research, please cite the following papers, [the algorithm](https://arxiv.org/pdf/2308.04526) and [the biological applications and software](https://www.biorxiv.org/content/10.1101/2024.09.02.610652).

```
@inproceedings{bragantini2024ucmtracking,
  title={Large-scale multi-hypotheses cell tracking using ultrametric contours maps},
  author={Bragantini, Jord{\~a}o and Lange, Merlin and Royer, Lo{\"\i}c},
  booktitle={European Conference on Computer Vision},
  pages={36--54},
  year={2024},
  organization={Springer}
}

@article{bragantini2024ultrack,
  title={Ultrack: pushing the limits of cell tracking across biological scales},
  author={Bragantini, Jord{~a}o and Theodoro, Ilan and Zhao, Xiang and Huijben, Teun APM and Hirata-Miyasaki, Eduardo and VijayKumar, Shruthi and Balasubramanian, Akilandeswari and Lao, Tiger and Agrawal, Richa and Xiao, Sheng and others},
  journal={bioRxiv},
  pages={2024--09},
  year={2024},
  publisher={Cold Spring Harbor Laboratory}
}
```

And the respective auxiliary methods according to your use, a non-exhaustive list being:
- [napari](https://github.com/Napari/napari)
- [cellpose](https://github.com/MouseLand/cellpose)
- [stardist](https://github.com/stardist/stardist)
- [fiji](https://fiji.sc)

## Acknowledgements

We acknowledge the contributions of the community and specific individuals. Detailed acknowledgments can be found in our documentation.
