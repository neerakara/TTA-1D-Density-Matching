#!/bin/bash
for tr_dataset in 'HCPT1' 'UMC' 'CSF' 'site2' # 'RUNMC' 'HCPT1' 'UMC' 'CSF' 'site2'
do
    for ae_features in 'xn' 'f1' 'f2' 'f3' 'y'
    do
        filename=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/train_ae.py
        tr_run_number=1
        bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename $tr_dataset $tr_run_number $ae_features
    done
done
