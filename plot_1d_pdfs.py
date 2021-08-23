# ==================================================================
# import 
# ==================================================================
import logging
import os.path
import numpy as np
import utils
import config.system_paths as sys_config
import config.params as exp_config
import argparse
import utils_vis

# ==================================================================
# setup logging
# ==================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ==================================================================
# parse arguments
# ==================================================================
parser = argparse.ArgumentParser(prog = 'PROG')
# Training dataset and run number
parser.add_argument('--train_dataset', default = "RUNMC") # RUNMC
parser.add_argument('--tr_run_number', type = int, default = 1) # 1 / 
# Batch settings
parser.add_argument('--feature_subsampling_factor', type = int, default = 16) # 1 / 8 / 16
parser.add_argument('--features_randomized', type = int, default = 1) # 1 / 0
# KDE hyperparameters
parser.add_argument('--KDE_ALPHA', type = float, default = 100.0) # 10.0 / 100.0 / 1000.0
# parse arguments
args = parser.parse_args()

# ================================================================
# set dataset dependent hyperparameters
# ================================================================
dataset_params = exp_config.get_dataset_dependent_params(args.train_dataset) 
b_size = dataset_params[9]

# ================================================================
# Setup directories
# ================================================================
expname_i2l = 'tr' + args.train_dataset + '_r' + str(args.tr_run_number) + '/' + 'i2i2l/'
log_dir = sys_config.project_root + 'log_dir/' + expname_i2l
logging.info('SD training directory: %s' %log_dir)
log_dir_pdfs = log_dir + 'onedpdfs/'

kde_g1_params = [-3.0, 3.0, 0.1]
kde_g2_params = [-5.0, 5.0, 0.1]
kde_g3_params = [0.0, 1.0, 0.01]
sd_gaussians_fname = exp_config.make_sd_gaussian_names(log_dir_pdfs, args)
sd_kdes_fname_g1 = log_dir_pdfs + exp_config.make_sd_kde_name(b_size, args.KDE_ALPHA, kde_g1_params) + '_g1.npy'
sd_kdes_fname_g2 = log_dir_pdfs + exp_config.make_sd_kde_name(b_size, args.KDE_ALPHA, kde_g2_params) + '_g2.npy'
sd_kdes_fname_g3 = log_dir_pdfs + exp_config.make_sd_kde_name(b_size, args.KDE_ALPHA, kde_g3_params) + '_g3.npy'

sd_gaussians = np.load(sd_gaussians_fname)
sd_kdes_g1 = np.load(sd_kdes_fname_g1)
sd_kdes_g2 = np.load(sd_kdes_fname_g2)
sd_kdes_g3 = np.load(sd_kdes_fname_g3)

# list of channels to be visualized
g1_layers = [0, # layer 1_1
             32, # 2_1
             96, # 3_1
             224, # 4_1
             480, # 5_1
             608, # 6_1
             672] # 7_1
g2_layers = [0] # 7_2
g3_layers = [0] # seg probs
num_channels = 10

s = 0 # subject 0
for l in g1_layers:                
    for c in range(num_channels):
        utils_vis.save_1d_pdfs(sd_gaussians[s, l+c, :],
                               sd_kdes_g1[s, l+c, :],
                               kde_g1_params,
                               log_dir_pdfs + 's0_c' + str(l+c) + '.png')
for l in g2_layers:                
    for c in range(num_channels):
        utils_vis.save_1d_pdfs(sd_gaussians[s, 688+l+c, :],
                               sd_kdes_g2[s, l+c, :],
                               kde_g2_params,
                               log_dir_pdfs + 's0_c' + str(688+l+c) + '.png')
for l in g3_layers:                
    for c in range(num_channels):
        utils_vis.save_1d_pdfs(sd_gaussians[s, 704+l+c, :],
                               sd_kdes_g3[s, l+c, :],
                               kde_g3_params,
                               log_dir_pdfs + 's0_c' + str(704+l+c) + '.png')