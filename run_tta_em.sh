#!/bin/bash

filename=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/tta_em.py
tr_runnum=5
bsize=8

for ts_dataset in 'NUHS' # 'BMC' 'USZ' 'UCL' 'HK' 'BIDMC' 'UHE' 'HVHD' 'site1' 'site3' 'site4' 'NUHS' 'CALTECH'
do   
    
    # run transfer learning for each test dataset, with the appropriate
    # 1. source dataset 
    # 2. cross validation fold number
    # 3. batch size (2 for site1 in spine datasets, 8 for all others)
    if [ "$ts_dataset" == "BMC" -o "$ts_dataset" == "USZ" ]; then
        for sub in $(seq 0 9)
        do
            bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'RUNMC' $tr_runnum $ts_dataset 1 $sub $bsize
        done
    
    elif [ "$ts_dataset" == "UCL" -o "$ts_dataset" == "HK" -o "$ts_dataset" == "BIDMC" ]; then
        for ts_cv in 1 2
        do
            for sub in $(seq 0 4)
            do
                bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'RUNMC' $tr_runnum $ts_dataset $ts_cv $sub $bsize
            done
        done
    
    elif [ "$ts_dataset" == "UHE" -o "$ts_dataset" == "HVHD" ]; then
        for sub in $(seq 0 19)
        do
            bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'CSF' $tr_runnum $ts_dataset 1 $sub $bsize
        done
    
    elif [ "$ts_dataset" == "site3" -o "$ts_dataset" == "site4" ]; then
        for ts_cv in 1 2 3
        do
            for sub in $(seq 0 2)
            do
                bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'site2' $tr_runnum $ts_dataset $ts_cv $sub $bsize
            done
        done
    
    elif [ "$ts_dataset" == "site1" ]; then
        for ts_cv in 1 2 3
        do
            for sub in $(seq 0 2)
            do
                bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'site2' $tr_runnum $ts_dataset $ts_cv $sub 2
            done
        done
    
    elif [ "$ts_dataset" == "NUHS" ]; then
        for ts_cv in 1 2
        do
            for sub in $(seq 0 4)
            do
                bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'UMC' $tr_runnum $ts_dataset $ts_cv $sub $bsize
            done
        done
    
    elif [ "$ts_dataset" == "CALTECH" ]; then
        for sub in $(seq 0 9)
        do
            bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'HCPT1' $tr_runnum $ts_dataset 1 $sub $bsize
        done
    
    else
        echo "I do not recognize this test dataset."
    fi

done