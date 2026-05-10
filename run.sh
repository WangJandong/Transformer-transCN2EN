#!/bin/bash
# Usage:  bash run.sh [extra args for train.py]
# Activates the insta360 conda env and starts training.

set -e

source /home/wjd/miniforge3/etc/profile.d/conda.sh
conda activate insta360
echo "Using: $(which python)"
python train.py "$@"