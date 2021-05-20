#!/bin/bash
for test_dataset in 'PROMISE' # 'USZ'
do

    for match_moments in 'firsttwo_kl' # 'all' 'all_std' 'all_std_log' 'all_std_log_multiply' # 'all_std_log_multiply' # 'all_std_log' # 'first' 'firsttwo_kl' 'firsttwo' # 'all' 'CF' 'CF_mag' 
    do   
        for tta_vars in 'bn' # 'bn'
        do  
            for sub in $(seq 0 19) # 0 2 3 4 6 7 8 10 12 13 14 15 16 17 18 19 # 1 5 9 11  #  # 12 15 16 19 # 29, 46, 11, 34, 9 $(seq 1 4)
            do   
                # n=20
                # norm=1
                bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $test_dataset $sub $tta_vars $match_moments
            done                     
        done    
    done      
done
