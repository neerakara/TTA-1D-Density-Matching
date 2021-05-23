#!/bin/bash

## Activate virtual environment
source /cluster/home/nkarani/envs/tf_v1_12/bin/activate

## Load compatible cuda and cudnn
module load cuda/9.0.176 cudnn/7.1.4

## EXECUTION OF PYTHON CODE on GPU:
# bsub -R "rusage[mem=4000,ngpus_excl_p=1]" -W 11:59 -oo /cluster/home/nkarani/logs/ python /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/simulated_domain_shifts_adaBN_v2.py \
# --adaBN $1 \
# --ds_order $2 \
# --num_total_iterations $3 \
# --normalize_after_ds $4 \
# --test_sub_num $5

#  select[gpu_mtotal0>=10240] --> GeForce RTX 2080 Ti major
#  select[gpu_mtotal0>=23000] --> TITAN RTX major

# bsub -R "rusage[mem=4000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 01:59 -oo /cluster/home/nkarani/logs/ python /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/update_i2i_with_pdf_matching.py \
# --test_dataset $1 --PROMISE_SUB_DATASET $2 # --test_sub_num $2 --tta_vars $3 --match_moments $4 --match_with_sd $5

bsub -R "rusage[mem=4000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 01:59 -oo /cluster/home/nkarani/logs/ python /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/evaluate.py \
--test_dataset $1 --PROMISE_SUB_DATASET $2 # --tta_vars $2 --match_moments $3 --match_with_sd $4

echo "Reached end of job file."
