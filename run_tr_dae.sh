#!/bin/bash
for tr_dataset in 'UMC' # 'HCPT1' 'UMC' 'CSF' 'site2' # 'RUNMC' 'HCPT1' 'UMC' 'CSF' 'site2'
do
    filename=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/train_dae.py
    tr_run_number=3
    bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename $tr_dataset $tr_run_number
done
