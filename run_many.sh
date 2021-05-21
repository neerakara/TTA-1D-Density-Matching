#!/bin/bash
for test_dataset in 'PROMISE' 'USZ'
do
    for match_moments in 'firsttwo_kl' # all / all_kl / all_std / all_std_log / all_std_log_multiply / first / firsttwo / firsttwo_kl / CF / CF_mag
    do   
        for tta_vars in 'norm' # norm / bn
        do  
            for sub in $(seq 0 19) 
            do   
                # n=20
                # norm=1
                match=4
                bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $test_dataset $sub $tta_vars $match_moments $match
            done                     
        done    
    done      
done
