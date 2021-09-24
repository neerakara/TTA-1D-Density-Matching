#!/bin/bash

filename=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/baseline_tl.py
tr_runnum=1
tl_vars='ALL'
tl_runnum=1

for ts_dataset in 'BMC' 'USZ' 'UCL' 'HK' 'BIDMC' 'UHE' 'HVHD' 'site1' 'site3' 'site4' 'NUHS' 'CALTECH'
do   
    
    # run transfer learning for each test dataset, with the appropriate source dataset and the appropriate cross validation fold number
    if [ "$ts_dataset" == "BMC" -o "$ts_dataset" == "USZ" ]; then
        bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'RUNMC' $tr_runnum $ts_dataset 1 $tl_vars $tl_runnum
    elif [ "$ts_dataset" == "UCL" -o "$ts_dataset" == "HK" -o "$ts_dataset" == "BIDMC" ]; then
        bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'RUNMC' $tr_runnum $ts_dataset 1 $tl_vars $tl_runnum
        bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'RUNMC' $tr_runnum $ts_dataset 2 $tl_vars $tl_runnum
    elif [ "$ts_dataset" == "UHE" -o "$ts_dataset" == "HVHD" ]; then
        bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'CSF' $tr_runnum $ts_dataset 1 $tl_vars $tl_runnum
    elif [ "$ts_dataset" == "site1" -o "$ts_dataset" == "site3" -o "$ts_dataset" == "site4" ]; then
        bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'site2' $tr_runnum $ts_dataset 1 $tl_vars $tl_runnum
        bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'site2' $tr_runnum $ts_dataset 2 $tl_vars $tl_runnum
        bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'site2' $tr_runnum $ts_dataset 3 $tl_vars $tl_runnum
    elif [ "$ts_dataset" == "NUHS" ]; then
        bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'UMC' $tr_runnum $ts_dataset 1 $tl_vars $tl_runnum
        bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'UMC' $tr_runnum $ts_dataset 2 $tl_vars $tl_runnum
    elif [ "$ts_dataset" == "CALTECH" ]; then
        bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'HCPT1' $tr_runnum $ts_dataset 1 $tl_vars $tl_runnum
    else
        echo "I do not recognize this test dataset."
    fi

done