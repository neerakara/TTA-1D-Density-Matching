#!/bin/bash

## Activate virtual environment
source /cluster/home/nkarani/envs/tf_v1_12/bin/activate

## Load compatible cuda and cudnn
module load cuda/9.0.176 cudnn/7.1.4

## EXECUTION OF PYTHON CODE on GPU:
#  select[gpu_mtotal0>=10240] --> GeForce RTX 2080 Ti major
#  select[gpu_mtotal0>=23000] --> TITAN RTX major

# list all files that can be asked to run on the GPU cluster
# Training on the source domain with data augmentation (baseline)
fname_sd_tr=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/train_i2l.py
# Transfer learning - finetuning using labelled images in the target domain (benchmark)
fname_tl=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/baseline_tl.py
# Test-Time Adaptation using Entropy Minimization (Wang et al, ICLR 2021)
fname_tta_em=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/tta_em.py
# Test-Time Adaptation using Autoencoders (He et al, MedIA 2021)
fname_tta_ae=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/tta_ae.py
fname_train_ae=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/train_ae.py
# Test-Time Adaptation using Denoising Autoencoders (Karani et al, MedIA 2021)
fname_tta_dae=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/tta_dae.py
fname_train_dae=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/train_dae.py
# Test-Time Adaptation using Fields of Experts (Ours)
fname_tta_foe=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/tta_foe.py
fname_compute_cnn_gaussians_foe=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/save_cnn_1d_gaussians.py
fname_compute_pca_gaussians_foe=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/save_pca_1d_gaussians.py
fname_compute_cnn_kdes_foe=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/save_cnn_1d_kdes.py
fname_compute_pca_kdes_foe=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/save_pca_1d_kdes.py
# Evaluate
fname_eval=/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/evaluate.py

# run the requested file with the appropriate arguments
if [ "$1" == "$fname_sd_tr" ]; then
    bsub -R "rusage[mem=4000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 23:59 -oo /cluster/home/nkarani/logs/ python $1 \
    --train_dataset $2 \
    --tr_run_number $3

elif [ "$1" == "$fname_tl" ]; then
    bsub -R "rusage[mem=4000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 05:59 -oo /cluster/home/nkarani/logs/ python $1 \
    --train_dataset $2 \
    --tr_run_number $3 \
    --test_dataset $4 \
    --test_cv_fold_num $5 \
    --TL_VARS $6 \
    --tl_runnum $7 \
    --batch_size $8

elif [ "$1" == "$fname_tta_em" ]; then
    bsub -R "rusage[mem=4000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 01:59 -oo /cluster/home/nkarani/logs/ python $1 \
    --train_dataset $2 \
    --tr_run_number $3 \
    --test_dataset $4 \
    --test_cv_fold_num $5 \
    --test_sub_num $6 \
    --b_size $7

elif [ "$1" == "$fname_tta_ae" ]; then
    bsub -R "rusage[mem=4000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 01:59 -oo /cluster/home/nkarani/logs/ python $1 \
    --train_dataset $2 \
    --test_dataset $3 \
    --test_cv_fold_num $4 \
    --test_sub_num $5 \
    --lambda_spectral $6 \
    --whichAEs $7

elif [ "$1" == "$fname_train_ae" ]; then
    bsub -R "rusage[mem=4000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 05:59 -oo /cluster/home/nkarani/logs/ python $1 \
    --train_dataset $2 \
    --tr_run_number $3 \
    --ae_features $4

elif [ "$1" == "$fname_tta_foe" ]; then
    bsub -R "rusage[mem=4000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 01:59 -oo /cluster/home/nkarani/logs/ python $1 \
    --train_dataset $2 \
    --tr_run_number $3 \
    --test_dataset $4 \
    --test_cv_fold_num $5 \
    --test_sub_num $6 \
    --TTA_VARS $7 \
    --PDF_TYPE $8 \
    --KDE_ALPHA $9 \
    --feature_subsampling_factor ${10} \
    --features_randomized ${11} \
    --b_size ${12} \
    --tta_learning_rate ${13} \
    --PCA_LAMBDA ${14} \
    --PCA_STRIDE ${15}

elif [ "$1" == "$fname_compute_cnn_gaussians_foe" ]; then
    bsub -R "rusage[mem=4000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 00:59 -oo /cluster/home/nkarani/logs/ python $1 \
    --train_dataset $2 \
    --tr_run_number $3 \
    --feature_subsampling_factor $4 \
    --features_randomized $5

elif [ "$1" == "$fname_compute_pca_gaussians_foe" ]; then
    bsub -R "rusage[mem=4000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 00:59 -oo /cluster/home/nkarani/logs/ python $1 \
    --train_dataset $2 \
    --tr_run_number $3

elif [ "$1" == "$fname_eval" ]; then
    bsub -R "rusage[mem=4000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 00:59 -oo /cluster/home/nkarani/logs/ python $1 \
    --train_dataset $2 \
    --tr_run_number $3 \
    --test_dataset $4 \
    --test_cv_fold_num $5 \
    --NORMALIZE $6 \
    --TRANSFER $7 \
    --b_size $8 \
    --tta_learning_rate $9 \
    --stopping_criterion ${10} \
    --TTA_VARS ${11} \
    --tta_method ${12} \
    --lambda_spectral ${13} \
    --whichAEs ${14} \
    --PDF_TYPE ${15} \
    --PCA_LAMBDA ${16} \
    --PCA_STRIDE ${17} \
    --feature_subsampling_factor ${18} \
    --features_randomized ${19} \
    --match_with_sd ${20} \
    --KDE_ALPHA ${21}

elif [ "$1" == "$fname_train_ae" ]; then
    bsub -R "rusage[mem=4000,ngpus_excl_p=1]" -R "select[gpu_mtotal0>=10240]" -W 05:59 -oo /cluster/home/nkarani/logs/ python $1 \
    --train_dataset $2 \
    --ae_features $3

else
    echo "I don't recognize the file that you want me run."
fi

echo "Reached end of job file."