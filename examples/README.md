# Ultrack's usage examples

Additional packages might be required.
Therefore, conda environment files are provided, they can be installed using:

```
conda env create -f <environment-file.yml>
conda activate <your new env>
```

The existing examples are:

- [multi_color_ensemble](./multi_color_ensemble): Multi-colored cytoplasm cell tracking using Cellpose and Watershed segmentation ensemble. Data provided by [The Lammerding Lab ](https://lammerding.wicmb.cornell.edu/).
- [flow_field_3d](./flow_field_3d): Tracking demo on a cartographic projection of Tribolium Castaneum embryo from the [cell-tracking challenge](http://celltrackingchallenge.net/3d-datasets/), using a flow field estimation to assist tracking of motile cells.
- [stardist_2d](./stardist_2d): Tracking demo on HeLa GPF nuclei from the [cell-tracking challenge](http://celltrackingchallenge.net/2d-datasets/) using Stardist 2D fluorescence images pre-trained model.
- [zebrahub](./zebrahub/): Tracking demo on zebrafish tail data from the [zebrahub](https://zebrahub.ds.czbiohub.org/) acquired with [DaXi](https://www.nature.com/articles/s41592-022-01417-2) using Ultrack's image processing helper functions.
- [neuromast_plantseg](./neuromast_plantseg/): Tracking demo membrane-labeled zebrafish neuromast from [Jacobo Group of CZ Biohub](https://www.czbiohub.org/jacobo/) using [PlantSeg's](https://github.com/hci-unihd/plant-seg) membrane detection model.
- [micro_sam](./micro_sam/): Tracking demo with [MicroSAM](https://github.com/computational-cell-analytics/micro-sam) instance segmentation package.


# Development Notes

To run all the examples and update the notebooks in headless mode, run:

```bash
bash refresh_examples.sh
```
