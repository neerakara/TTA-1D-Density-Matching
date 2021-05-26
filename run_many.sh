#!/bin/bash
for test_dataset in 'PROMISE' 'USZ' # USZ' # 'CALTECH' 'STANFORD' 'HCPT2'
do
    for train_dataset in 'NCI' # all / all_kl / all_std / all_std_log / all_std_log_multiply / first / firsttwo / firsttwo_kl / CF / CF_mag
    do   
        for tta_vars in 'norm' # norm / bn
        do  
            for sub in $(seq 3 19) 
            do   
                # n=20
                # norm=1
                # match=2
                alpha=100.0
                bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $train_dataset $test_dataset $sub $alpha # $tta_vars $match_moments $match
            done                     
        done    
    done      
done
