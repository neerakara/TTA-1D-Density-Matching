#!/bin/bash

## Activate virtual environment
source /cluster/home/nkarani/envs/tf_v1_12/bin/activate

## Load compatible cuda and cudnn
module load cuda/9.0.176 cudnn/7.1.4

## EXECUTION OF PYTHON CODE on GPU:
#  select[gpu_mtotal0>=10240] --> GeForce RTX 2080 Ti major
#  select[gpu_mtotal0>=23000] --> TITAN RTX major

bsub -R "rusage[mem=4000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 11:59 -oo /cluster/home/nkarani/logs/ python /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/save_1d_gaussians.py
# --test_dataset $1 --test_cv_fold_num $2

echo "Reached end of job file."
