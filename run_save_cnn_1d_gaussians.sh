#!/bin/bash
for tr_dataset in 'RUNMC' 'CSF' 'UMC' 'site2' 'HCPT1'
do
    filename=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/save_cnn_1d_gaussians.py
    tr_runnum=1
    subsample=1
    randomize=0
    bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename $tr_dataset $tr_runnum $subsample $randomize
done
