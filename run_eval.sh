#!/bin/bash

for train_dataset in 'HCPT1' # 'KDE' # 
do
    for test_dataset in 'CALTECH' # 'HVHD' # 'CALTECH' # 'UHE' 'HVHD' # 'BMC' 'USZ'
    do
        for pcalambda in 1.0 # 1.0 # 0.1 # 0.05 0.1 # 0.25 # 0.5 # 0.1 0.5
        do
            for cv in 1
            do   
                # bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $test_dataset $cv $train_dataset $pcalambda
                for sub in $(seq 0 9)
                do
                    bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $test_dataset $cv $train_dataset $sub $pcalambda
                done
            done
        done      
    done
done