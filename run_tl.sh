#!/bin/bash
for ts_dataset in 'CALTECH' # 'NUHS' # 'site1' 'site3' 'site4' # 'UHE' 'HVHD' # 'UCL' 'HK' 'BIDMC' # 'BMC' 'USZ'
do
    for ts_cv in 1
    do
        tr_dataset='HCPT1'
        tr_cv=1
        bash /cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/run.sh $tr_dataset $tr_cv $ts_dataset $ts_cv
    done
done
