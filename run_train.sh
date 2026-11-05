#!/bin/bash
source /home/mehtaj3/miniforge3/etc/profile.d/conda.sh
conda activate risl
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"
python train_car_classifier.py --samples 2500 --batch-size 64 --epochs 15 2>&1 | tee training.log

