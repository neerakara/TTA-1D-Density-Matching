# ==================================================================
# import 
# ==================================================================
import logging
import os.path
import tensorflow as tf
import numpy as np
import utils
import utils_vis
import utils_kde
import model as model
import config.system_paths as sys_config
import config.params as exp_config
from skimage.transform import rescale
import sklearn.metrics as met
import argparse
from sklearn.decomposition import PCA
import pickle as pk

# ==================================================================
# setup logging
# ==================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ==================================================================
# parse arguments
# ==================================================================
parser = argparse.ArgumentParser(prog = 'PROG')
# Training dataset and run number
parser.add_argument('--train_dataset', default = "NCI") # NCI / HCPT1
parser.add_argument('--tr_run_number', type = int, default = 1) # 1 / 
# Test dataset 
parser.add_argument('--test_dataset', default = "PROMISE") # PROMISE / USZ / CALTECH / STANFORD / HCPT2
parser.add_argument('--NORMALIZE', type = int, default = 1) # 1 / 0
# TTA options
parser.add_argument('--tta_string', default = "TTA/")
parser.add_argument('--adaBN', type = int, default = 0) # 0 to 1
# Whether to compute KDE or not?
parser.add_argument('--KDE', type = int, default = 1) # 0 to 1
parser.add_argument('--alpha', type = float, default = 100.0) # 10.0 / 100.0 / 1000.0
# Which vars to adapt?
parser.add_argument('--tta_vars', default = "NORM") # BN / NORM
# How many moments to match and how?
parser.add_argument('--match_moments', default = "All_KL") # Gaussian_KL / All_KL / All_CF_L2
parser.add_argument('--before_or_after_bn', default = "AFTER") # AFTER / BEFORE
# PCA settings
parser.add_argument('--patch_size', type = int, default = 128) # 32 / 64 / 128
parser.add_argument('--pca_stride', type = int, default = 32) # 64 / 128
parser.add_argument('--pca_layer', default = 'logits') # layer_7_2 / logits
parser.add_argument('--pca_channel', type = int, default = 0) # 0 / 1 .. 15
parser.add_argument('--PCA_LATENT_DIM', type = int, default = 10) # 10 / 50
parser.add_argument('--pca_kde_alpha', type = float, default = 1.0) # 0.1 / 1.0 / 10.0
# Batch settings
parser.add_argument('--b_size', type = int, default = 16) # 1 / 2 / 4 (requires 24G GPU)
parser.add_argument('--feature_subsampling_factor', type = int, default = 16) # 1 / 4
parser.add_argument('--features_randomized', type = int, default = 1) # 1 / 0
parser.add_argument('--use_logits_for_TTA', type = int, default = 0) # 1 / 0
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
# set dataset dependent hyperparameters
# ================================================================
dataset_params = exp_config.get_dataset_dependent_params(args.train_dataset, args.test_dataset) 
image_size = dataset_params[0]
nlabels = dataset_params[1]
target_resolution = dataset_params[2]
image_depth_tr = dataset_params[3]
image_depth_ts = dataset_params[4]
whole_gland_results = dataset_params[5]

# ================================================================
# Setup directories for this run
# ================================================================
expname_i2l = 'tr' + args.train_dataset + '_r' + str(args.tr_run_number) + '/' + 'i2i2l/'
log_dir_sd = sys_config.project_root + 'log_dir/' + expname_i2l

# ==================================================================
# ==================================================================
def main():

    # ===================================
    # load training images
    # ===================================
    imtr, gttr, orig_data_siz_z_train, num_train_subjects = utils.load_training_data(args.train_dataset,
                                                                                     image_size,
                                                                                     target_resolution)
     
    # ===================================
    # read the test images
    # ===================================
    test_dataset_name = args.test_dataset
    loaded_test_data = utils.load_testing_data(test_dataset_name,
                                               image_size,
                                               target_resolution,
                                               image_depth_ts)

    imts = loaded_test_data[0]
    gtts = loaded_test_data[1]
    orig_data_res_x = loaded_test_data[2]
    orig_data_res_y = loaded_test_data[3]
    orig_data_res_z = loaded_test_data[4]
    orig_data_siz_x = loaded_test_data[5]
    orig_data_siz_y = loaded_test_data[6]
    orig_data_siz_z = loaded_test_data[7]
    name_test_subjects = loaded_test_data[8]
    num_test_subjects = loaded_test_data[9]
    ids = loaded_test_data[10]
        
    # extract the single test volume
    for sub_num in range(10):
        print('================' + test_dataset_name + ', subject: ' + str(sub_num))
        subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])
        subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])
        test_image = imts[subject_id_start_slice:subject_id_end_slice,:,:]  
        test_image_gt = gtts[subject_id_start_slice:subject_id_end_slice,:,:]  
        test_image_gt = test_image_gt.astype(np.uint8)

        # print('test image shape after preproc: ' + str(test_image.shape))
        # print('test image shape before preproc: ' + str(orig_data_siz_x[sub_num]) + ', ' + str(orig_data_siz_y[sub_num]) + ', ' + str(orig_data_siz_z[sub_num]))
        # print('test image resolution before preproc: ' + str(orig_data_res_x[sub_num]) + ', ' + str(orig_data_res_y[sub_num]) + ', ' + str(orig_data_res_z[sub_num]))

        nxhat = orig_data_siz_x[sub_num] * orig_data_res_x[sub_num] / target_resolution[0]
        nyhat = orig_data_siz_y[sub_num] * orig_data_res_y[sub_num] / target_resolution[1]
        # print('test image shape after preproc step 1 (rescaling to match resolution to SD images): ' + str(nxhat) + ', ' + str(nyhat))

        delta_x = np.maximum(0, test_image.shape[1] - nxhat.astype(np.uint16))
        delta_y = np.maximum(0, test_image.shape[2] - nyhat.astype(np.uint16))
        
        if test_dataset_name == 'USZ':
            print('Number of zeros padded on both ends of the x-axis: ' + str(delta_y // 2))
            print('Number of zeros padded on both ends of the y-axis: ' + str(delta_x // 2))
        else:
            print('Number of zeros padded on both ends of the x-axis: ' + str(delta_x // 2))
            print('Number of zeros padded on both ends of the y-axis: ' + str(delta_y // 2))

        print("Number of unique values in the first 10 rows: " + str(len(np.unique(test_image[test_image.shape[0]//2, :10, :]))))
        print("Number of unique values in the first 10 columns: " + str(len(np.unique(test_image[test_image.shape[0]//2, :, :10]))))


# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()