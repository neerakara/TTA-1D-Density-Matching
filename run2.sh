#!/bin/bash

for pdftype in 'KDE' # 'GAUSSIAN' 
do
    for test_dataset in 'UCL' 'BIDMC' 'HK' # 'BMC' 'USZ'
    do
        for pcalambda in 0.1 # 0.25 # 0.5 # 0.1 0.5
        do
            for cv in 1 2
            do   
                # bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $test_dataset $cv $pdftype $pcalambda
                for sub in $(seq 0 4)
                do
                    bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $test_dataset $cv $pdftype $sub $pcalambda
                done
            done
        done      
    done
done
