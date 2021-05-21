# ==================================================================
# import 
# ==================================================================
import logging
import os.path
import numpy as np
import config.system as sys_config
import argparse

# read arguments
parser = argparse.ArgumentParser(prog = 'PROG')
parser.add_argument('--test_dataset', default = "PROMISE") # USZ / PROMISE
parser.add_argument('--tta_vars', default = "norm") # bn / norm
parser.add_argument('--match_moments', default = "all_kl") # first / firsttwo / all
parser.add_argument('--b_size', type = int, default = 16) # 1 / 2 / 4 (requires 24G GPU)
parser.add_argument('--batch_randomized', type = int, default = 1) # 1 / 0
parser.add_argument('--feature_subsampling_factor', type = int, default = 8) # 1 / 4
parser.add_argument('--features_randomized', type = int, default = 1) # 1 / 0
parser.add_argument('--match_with_sd', type = int, default = 3) # 1 / 2 / 3
args = parser.parse_args()

# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments import i2i as exp_config
log_root = sys_config.project_root + 'log_dir/'

log_dir_sd = log_root + exp_config.expname_i2l
exp_str = exp_config.tta_string + 'tta_vars_' + args.tta_vars 
exp_str = exp_str + '/moments_' + args.match_moments
exp_str = exp_str + '_bsize' + str(args.b_size)
exp_str = exp_str + '_rand' + str(args.batch_randomized)
exp_str = exp_str + '_fs' + str(args.feature_subsampling_factor)
exp_str = exp_str + '_rand' + str(args.features_randomized)
exp_str = exp_str + '_sd_match' + str(args.match_with_sd)
exp_str = exp_str + '/' 
log_dir_tta = log_dir_sd + exp_str

test_dataset_name = args.test_dataset

if test_dataset_name == 'PROMISE':
    
    if exp_config.normalize == True:
        with open(log_dir_tta + test_dataset_name + '_test_whole_gland.txt', "r") as f:
            lines = f.readlines()
    else:
        with open(log_dir_sd + test_dataset_name + '_test_whole_gland.txt', "r") as f:
            lines = f.readlines()
    
    pat_id = []
    dice = []
    for count in range(2, 22):
        line = lines[count]
        pat_id.append(int(line[4:6]))
        dice.append(float(line[46:46+line[46:].find(',')]))

    pat_id = np.array(pat_id)
    dice = np.array(dice)
    results = np.stack((pat_id, dice))

    sorted_results = np.stack((np.sort(results[0,:]),
                               results[1, np.argsort(results[0,:])]))

    print('========== sorted results ==========')
    for c in range(1, sorted_results.shape[1]):
        print(str(sorted_results[0,c]) + ',' + str(sorted_results[1,c]))
        if c == 9:
            print(str(sorted_results[0,0]) + ',' + str(sorted_results[1,0]))
    print('====================================')
    print(lines[31])
    print('====================================')

elif test_dataset_name == 'USZ':
    
    if exp_config.normalize == True:
        with open(log_dir_tta + test_dataset_name + '_test_whole_gland.txt', "r") as f:
            lines = f.readlines()
    else:
        with open(log_dir_sd + test_dataset_name + '_test_whole_gland.txt', "r") as f:
            lines = f.readlines()
    
    pat_id = []
    dice = []
    for count in range(2, 22):
        line = lines[count]
        pat_id.append(int(line[6:line.find(':')]))
        line = line[line.find(':') + 39 : ]
        dice.append(float(line[:line.find(',')]))

    pat_id = np.array(pat_id)
    dice = np.array(dice)
    results = np.stack((pat_id, dice))

    sorted_results = np.stack((np.sort(results[0,:]),
                               results[1, np.argsort(results[0,:])]))

    print('========== sorted results ==========')
    for c in range(0, sorted_results.shape[1]):
        print(str(sorted_results[0,c]) + ',' + str(sorted_results[1,c]))
    print('====================================')
    print(lines[31])
    print('====================================')