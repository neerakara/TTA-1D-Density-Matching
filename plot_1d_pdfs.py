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
import tensorflow as tf

# ==================================================================
# setup logging
# ==================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ==================================================================
# parse arguments
# ==================================================================
parser = argparse.ArgumentParser(prog = 'PROG')
# Training dataset and run number
parser.add_argument('--train_dataset', default = "HCPT1") # RUNMC / CSF / UMC / HCPT1
parser.add_argument('--tr_run_number', type = int, default = 1) # 1 / 
parser.add_argument('--num_labels', type = int, default = 15) # 3 / 4 / 2 / 15
# Batch settings
parser.add_argument('--feature_subsampling_factor', type = int, default = 16) # 1 / 8 / 16
parser.add_argument('--features_randomized', type = int, default = 1) # 1 / 0
# KDE hyperparameters
parser.add_argument('--KDE_ALPHA', type = float, default = 10.0) # 10.0 / 100.0 / 1000.0
# PCA settings
parser.add_argument('--PCA_PSIZE', type = int, default = 16) # 32 / 64 / 128
parser.add_argument('--PCA_STRIDE', type = int, default = 8) # 64 / 128
parser.add_argument('--PCA_LAYER', default = 'layer_7_2') # layer_7_2 / logits / softmax
parser.add_argument('--PCA_LATENT_DIM', type = int, default = 10) # 10 / 50
parser.add_argument('--PCA_KDE_ALPHA', type = float, default = 10.0) # 0.1 / 1.0 / 10.0
parser.add_argument('--PCA_THRESHOLD', type = float, default = 0.8) # 0.8
# Save which pdfs
parser.add_argument('--save_cnn_pdfs', type = int, default = 1) # 1 / 0
parser.add_argument('--save_pca_pdfs', type = int, default = 0) # 1 / 0
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

# ==========================
# save CNN PDFs
# ==========================
if args.save_cnn_pdfs == 1:
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
                16, # 1_2
                32, # 2_1
                64, # 2_2
                96, # 3_1
                160, # 3_2
                224, # 4_1
                352, # 4_2
                480, # 5_1
                544, # 5_2
                608, # 6_1
                640, # 6_2
                672] # 7_1
    g2_layers = [0] # 7_2
    g3_layers = [0] # seg probs
    num_channels = 10

    cnn_kde_savedir = log_dir_pdfs + 'figures/kde_alpha_' + str(args.KDE_ALPHA) + '/'
    if not tf.gfile.Exists(cnn_kde_savedir):
        tf.gfile.MakeDirs(cnn_kde_savedir)
    s = 0 # subject 0

    for l in g1_layers:                
        for c in range(num_channels):
            utils_vis.save_1d_pdfs(sd_gaussians[s, l+c, :],
                                sd_kdes_g1[s, l+c, :],
                                kde_g1_params,
                                cnn_kde_savedir + 's0_c' + str(l+c) + '.png')
    for l in g2_layers:                
        for c in range(num_channels):
            utils_vis.save_1d_pdfs(sd_gaussians[s, 688+l+c, :],
                                sd_kdes_g2[s, l+c, :],
                                kde_g2_params,
                                cnn_kde_savedir + 's0_c' + str(688+l+c) + '.png')
    for l in g3_layers:                
        for c in range(args.num_labels):
            utils_vis.save_1d_pdfs(sd_gaussians[s, 704+l+c, :],
                                sd_kdes_g3[s, l+c, :],
                                kde_g3_params,
                                cnn_kde_savedir + 's0_c' + str(704+l+c) + '.png')

# ==========================
# save PCA PDFs
# ==========================
if args.save_pca_pdfs == 1:
    pca_dir = log_dir_pdfs + exp_config.make_pca_dir_name(args)
    pca_kde_savedir = pca_dir + 'figures/'
    if not tf.gfile.Exists(pca_kde_savedir):
        tf.gfile.MakeDirs(pca_kde_savedir)

    num_channels = 16
    s = 1 # subject 1
    kde_z_params = [-5.0, 5.0, 0.1]

    for channel in range(num_channels):

        latent_kdes_this_channel = np.load(pca_dir + 'c' + str(channel) + '.npy') # num_subjects, num_latent_dims, num_points

        for l in range(latent_kdes_this_channel.shape[1]):
            utils_vis.save_1d_pdfs_pca(latent_kdes_this_channel[s, l, :],
                                       kde_z_params,
                                       pca_kde_savedir + 's1_c' + str(channel) + '_z' + str(l) + '.png')