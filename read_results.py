# ==================================================================
# import 
# ==================================================================
import logging
import os.path
import numpy as np
import config.system_paths as sys_config
import config.params as exp_config
import argparse

# ==================================================================
# parse arguments
# ==================================================================
parser = argparse.ArgumentParser(prog = 'PROG')
# Training dataset and run number
parser.add_argument('--train_dataset', default = "NCI") # NCI / HCPT1
parser.add_argument('--tr_run_number', type = int, default = 1) # 1 / 
# Test dataset 
parser.add_argument('--test_dataset', default = "USZ") # PROMISE / USZ / CALTECH / STANFORD / HCPT2
parser.add_argument('--NORMALIZE', type = int, default = 1) # 1 / 0
# TTA options
parser.add_argument('--tta_string', default = "TTA/")
parser.add_argument('--adaBN', type = int, default = 0) # 0 to 1
# Whether to compute KDE or not?
parser.add_argument('--KDE', type = int, default = 1) # 0 to 1
parser.add_argument('--alpha', type = float, default = 100.0) # 10.0 / 100.0 / 1000.0
parser.add_argument('--KDE_Groups', type = int, default = 1) # 0 / 1
# PCA settings
parser.add_argument('--PCA_PSIZE', type = int, default = 16) # 16 / 32 / 64
parser.add_argument('--PCA_STRIDE', type = int, default = 8) # 8 / 16
parser.add_argument('--PCA_NUM_LATENTS', type = int, default = 10) # 5 / 10 / 50
parser.add_argument('--PCA_KDE_ALPHA', type = float, default = 10.0) # 10.0 / 100.0
parser.add_argument('--PCA_LAMBDA', type = float, default = 0.1) # 0.1 / 0.01
# Which vars to adapt?
parser.add_argument('--tta_vars', default = "NORM") # BN / NORM
# How many moments to match and how?
parser.add_argument('--match_moments', default = "All_KL") # Gaussian_KL / All_KL / All_CF_L2
parser.add_argument('--before_or_after_bn', default = "AFTER") # AFTER / BEFORE
# Batch settings
parser.add_argument('--b_size', type = int, default = 16) # 1 / 2 / 4 (requires 24G GPU)
parser.add_argument('--feature_subsampling_factor', type = int, default = 16) # 1 / 4
parser.add_argument('--features_randomized', type = int, default = 1) # 1 / 0
# Matching settings
parser.add_argument('--match_with_sd', type = int, default = 2) # 1 / 2 / 3 / 4
# Learning rate settings
parser.add_argument('--tta_learning_rate', type = float, default = 0.0001) # 0.001 / 0.0005 / 0.0001 
parser.add_argument('--tta_learning_sch', type = int, default = 0) # 0 / 1
# Re-INIT TTA vars?
parser.add_argument('--tta_init_from_scratch', type = int, default = 0) # 0 / 1
# SFDA options
parser.add_argument('--TTA_or_SFDA', default = "TTA") # TTA / SFDA
parser.add_argument('--PROMISE_SUB_DATASET', default = "RUNMC") # RUNMC / UCL / BIDMC / HK (doesn't matter for TTA)
# parse arguments
args = parser.parse_args()

# ================================================================
# Make the name for this TTA run
# ================================================================
exp_str = exp_config.make_tta_exp_name(args)

# ================================================================
# Setup directories for this run
# ================================================================
expname_i2l = 'tr' + args.train_dataset + '_r' + str(args.tr_run_number) + '/' + 'i2i2l/'
log_dir_sd = sys_config.project_root + 'log_dir/' + expname_i2l
log_dir_tta = log_dir_sd + exp_str

# ==================================================================
# ==================================================================
if args.TTA_or_SFDA == 'SFDA':
    if args.test_dataset == 'USZ':
        td_string = 'SFDA_' + args.test_dataset + '/'
    elif args.test_dataset == 'PROMISE':    
        td_string = 'SFDA_' + args.test_dataset + '_' + args.PROMISE_SUB_DATASET + '/'
    log_dir_tta = log_dir_tta + td_string

# ==================================================================
# ==================================================================
test_dataset_name = args.test_dataset

if test_dataset_name == 'PROMISE' or test_dataset_name == 'USZ':
    if args.NORMALIZE == 1:
        with open(log_dir_tta + test_dataset_name + '_test_whole_gland.txt', "r") as f:
            lines = f.readlines()
    else:
        with open(log_dir_sd + test_dataset_name + '_test_whole_gland.txt', "r") as f:
            lines = f.readlines()

elif test_dataset_name == 'CALTECH' or test_dataset_name == 'STANFORD':
    if args.NORMALIZE == 1:
        with open(log_dir_tta + test_dataset_name + '_test.txt', "r") as f:
            lines = f.readlines()
    else:
        with open(log_dir_sd + test_dataset_name + '_test.txt', "r") as f:
            lines = f.readlines()

# ==================================================================
# ==================================================================
pat_id = []
dice = []

for count in range(2, 22):
    line = lines[count]

    if test_dataset_name == 'PROMISE':
        pat_id.append(int(line[4:6]))
        dice.append(float(line[46:46+line[46:].find(',')]))
    elif test_dataset_name == 'USZ':
        pat_id.append(int(line[6:line.find(':')]))
        line = line[line.find(':') + 39 : ]
        dice.append(float(line[:line.find(',')]))

pat_id = np.array(pat_id)
dice = np.array(dice)
results = np.stack((pat_id, dice))
sorted_results = np.stack((np.sort(results[0,:]), results[1, np.argsort(results[0,:])]))

# ==================================================================
# ==================================================================
print('========== sorted results ==========')
if test_dataset_name == 'PROMISE':
    for c in range(1, sorted_results.shape[1]):
        print(str(sorted_results[0,c]) + ',' + str(sorted_results[1,c]))
        if c == 9:
            print(str(sorted_results[0,0]) + ',' + str(sorted_results[1,0]))

elif test_dataset_name == 'USZ':
    for c in range(0, sorted_results.shape[1]):
        print(str(sorted_results[0,c]) + ',' + str(sorted_results[1,c]))

print('====================================')
print(lines[31])
print('====================================')