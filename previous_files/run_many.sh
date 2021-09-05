#!/bin/bash
for test_dataset in 'USZ' # 'CALTECH' / 'STANFORD' / 'PROMISE' / 'USZ'
do
    for train_dataset in 'NCI' # NCI / HCPT1
    do   
        for match in 'All_KL' # 'All_KL' / All_KL_LEBESGUE / Gaussian_KL
        do  
            for sub in $(seq 1 19)
            do   
                # n=20
                # norm=1
                # kde=1
                alpha=100.0
                runnum=3
                bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $train_dataset $test_dataset $sub $alpha $runnum # $kde $match
            done                     
        done    
    done      
done
