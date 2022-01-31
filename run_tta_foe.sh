#!/bin/bash

filename=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/tta_foe.py
tr_runnum=1
ttavars='NORM' # AdaptAxAf / NORM / AdaptAx
pdftype='KDE' # GAUSSIAN / KDE
kdealpha=10.0 # relevant only if pdftype = 'KDE'
subsample=16
randomize=1
bsize=8
lr=0.0001 # 0.00001 / 0.0001
lam=0.1 # PCA LAMBDA
stride=8 # PCA STRIDE # for now, 2 for brain lesions, 8 for all others

for ts_dataset in 'USZ' 'UCL' 'HK' 'BIDMC' # 'BMC' 'USZ' 'UCL' 'HK' 'BIDMC' 'UHE' 'HVHD' 'site1' 'site3' 'site4' 'NUHS' 'CALTECH'
do   
    
    # run transfer learning for each test dataset, with the appropriate
    # 1. source dataset 
    # 2. cross validation fold number
    # 3. batch size (2 for site1 in spine datasets, 8 for all others)
    # 4. learning rate
    # 5. pca weight --> lambda 
    if [ "$ts_dataset" == "BMC" -o "$ts_dataset" == "USZ" ]; then
        for sub in $(seq 0 9)
        do
            bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'RUNMC' $tr_runnum $ts_dataset 1 $sub $ttavars $pdftype $kdealpha $subsample $randomize $bsize $lr $lam $stride
        done
    
    elif [ "$ts_dataset" == "UCL" -o "$ts_dataset" == "HK" -o "$ts_dataset" == "BIDMC" ]; then
        for ts_cv in 1 2
        do
            for sub in $(seq 0 4)
            do
                bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'RUNMC' $tr_runnum $ts_dataset $ts_cv $sub $ttavars $pdftype $kdealpha $subsample $randomize $bsize $lr $lam $stride
            done
        done
    
    elif [ "$ts_dataset" == "UHE" -o "$ts_dataset" == "HVHD" ]; then
        for sub in $(seq 0 19)
        do
            bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'CSF' $tr_runnum $ts_dataset 1 $sub $ttavars $pdftype $kdealpha $subsample $randomize $bsize $lr $lam $stride
        done
    
    elif [ "$ts_dataset" == "site3" -o "$ts_dataset" == "site4" ]; then
        for ts_cv in 1 2 3
        do
            for sub in $(seq 0 2)
            do
                bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'site2' $tr_runnum $ts_dataset $ts_cv $sub $ttavars $pdftype $kdealpha $subsample $randomize $bsize $lr $lam $stride
            done
        done
    
    elif [ "$ts_dataset" == "site1" ]; then
        for ts_cv in 1 2 3
        do
            for sub in $(seq 0 2)
            do
                bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'site2' $tr_runnum $ts_dataset $ts_cv $sub $ttavars $pdftype $kdealpha $subsample $randomize 2 $lr $lam $stride
            done
        done
    
    elif [ "$ts_dataset" == "NUHS" ]; then
        for ts_cv in 1 2
        do
            for sub in $(seq 0 4)
            do
                bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'UMC' $tr_runnum $ts_dataset $ts_cv $sub $ttavars $pdftype $kdealpha $subsample $randomize $bsize $lr $lam 2
            done
        done
    
    elif [ "$ts_dataset" == "CALTECH" ]; then
        for sub in $(seq 0 9)
        do
            bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename 'HCPT1' $tr_runnum $ts_dataset 1 $sub $ttavars $pdftype $kdealpha $subsample $randomize $bsize $lr $lam $stride
        done
    
    else
        echo "I do not recognize this test dataset."
    fi

done

# for train_dataset in 'RUNMC' # 'KDE' # 
# do
#     for test_dataset in 'UCL' 'HK' 'BIDMC' # 'site4' # 'HVHD' # 'CALTECH' # 'UHE' 'HVHD' # 'BMC' 'USZ'
#     do
#         for pcalambda in 1.0 # 1.0 # 1.0 # 0.1 # 0.05 0.1 # 0.25 # 0.5 # 0.1 0.5
#         do
#             for cv in 1 2
#             do   
#                 for sub in $(seq 0 4)
#                 do
#                     bsize=8
#                     lr=0.0001
#                     bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $train_dataset $test_dataset $cv $sub $pcalambda $bsize $lr
#                 done
#             done
#         done      
#     done
# done