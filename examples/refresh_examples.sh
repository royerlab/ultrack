#! /bin/bash

# subject to change depending on your conda setup
CONDA_SETUP_PATH=$HOME/miniconda3/etc/profile.d/conda.sh
UPDATE_JUPYTER="jupyter nbconvert --execute --to notebook --inplace"

function install () {
	source $CONDA_SETUP_PATH
	mamba env remove --name $1
	mamba env create --file $2/environment_gpu.yml
	conda activate $1
	pip install -e ..
}

# stardist
install ultrack-stardist stardist_2d
$UPDATE_JUPYTER stardist_2d/2d_tracking.ipynb

# zebrahub
install ultrack-zebrahub zebrahub
$UPDATE_JUPYTER zebrahub/zebrahub.ipynb

# flow field
install ultrack-flow-field flow_field_3d
$UPDATE_JUPYTER flow_field_3d/tribolium_cartograph.ipynb
