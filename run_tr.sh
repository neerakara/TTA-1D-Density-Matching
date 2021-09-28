#!/bin/bash
for tr_dataset in 'HCPT1' # 'RUNMC' 'CSF' 'UMC' 'site2'
do
    filename=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/train_i2l.py
    tr_runnum=1
    bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename $tr_dataset $tr_runnum
done
