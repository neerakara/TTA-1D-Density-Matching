#!/bin/bash

filename=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/evaluate.py
tr_runnum=1
eval_type=1 # 1 (SD + DA baseline) | 2 (TL benchmark) | 3 (TTA-EM) | 4 (TTA-AE) | 5 (TTA-DAE) | 6 (TTA-FoE)

transfer=0 # set =1 for transfer learning, 0 for rest
normalize=1 # set =1 for TTA, 0 for baseline and benchmark
# now, tta related parameters (don't matter for baseline and benchmark)
bsize=8
lr=0.0001
stop='best_loss'
tta_vars='NORM'
tta_method='FoE'
# matter only for TTA-AE
lam_spec=0.0
aes='xn_and_y'
# matter only for TTA-FoE
pdf='GAUSSIAN'
lam_pca=1.0
stride_pca=8
subsample=1 # 16 for KDE
randomize=0 # 1 for KDE
match=2 # 1 / 2
# matter only for TTA-FoE-KDE
kdealpha=10.0

# modify a subset of the parameters depending on the evaluation type
if [[ "$eval_type" -eq 1 ]]; then
    do_nothing=1 # evaluate with the default parameters set above
elif [[ "$eval_type" -eq 2 ]]; then
    transfer=1
else
    echo "I do not recognize this evaluation type."
fi
    
# run evaluation for each test dataset
for ts_dataset in 'UHE' 'HVHD' # 'BMC' 'USZ' 'UCL' 'HK' 'BIDMC' 'UHE' 'HVHD' 'site1' 'site3' 'site4' 'NUHS' 'CALTECH' # | 'RUNMC' 'CSF' 'UMC' 'site2' 'HCPT1'
do   
    if [ "$ts_dataset" == "BMC" -o "$ts_dataset" == "USZ" -o "$ts_dataset" == "RUNMC" ]; then
        tr_dataset='RUNMC'
        ts_cv=1
    elif [ "$ts_dataset" == "UCL" -o "$ts_dataset" == "HK" -o "$ts_dataset" == "BIDMC" ]; then
        tr_dataset='RUNMC'
        ts_cv=3
    elif [ "$ts_dataset" == "UHE" -o "$ts_dataset" == "HVHD" -o "$ts_dataset" == "CSF" ]; then
        tr_dataset='CSF'
        ts_cv=1
    elif [ "$ts_dataset" == "site1" -o "$ts_dataset" == "site3" -o "$ts_dataset" == "site4" ]; then
        tr_dataset='site2'
        ts_cv=4
    elif [ "$ts_dataset" == "site2" ]; then
        tr_dataset='site2'
        ts_cv=1
    elif [ "$ts_dataset" == "NUHS" ]; then
        tr_dataset='UMC'
        ts_cv=3
    elif [ "$ts_dataset" == "UMC" ]; then
        tr_dataset='UMC'
        ts_cv=1
    elif [ "$ts_dataset" == "CALTECH" -o "$ts_dataset" == "HCPT1" ]; then
        tr_dataset='HCPT1'
        ts_cv=1
    else
        echo "I do not recognize this test dataset."
    fi

    bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $filename $tr_dataset $tr_runnum $ts_dataset $ts_cv $normalize $transfer $bsize $lr $stop $tta_vars $tta_method $lam_spec $aes $pdf $lam_pca $stride_pca $subsample $randomize $match $kdealpha

done