# ==================================================================
# import 
# ==================================================================
import logging
import os.path
import tensorflow as tf
import numpy as np
import utils
import utils_vis
import model as model
import sklearn.metrics as met
import config.system as sys_config

import data.data_hcp as data_hcp
import data.data_abide as data_abide
import data.data_nci as data_nci
import data.data_promise as data_promise
import data.data_pirad_erc as data_pirad_erc

import argparse

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 

# ==================================================================
# setup logging
# ==================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments import i2i as exp_config
target_resolution = exp_config.target_resolution
image_size = exp_config.image_size
nlabels = exp_config.nlabels

# ==================================================================
# Read PROMISE
# ==================================================================
# data_pros = data_promise.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_promise,
#                                                         preprocessing_folder = sys_config.preproc_folder_promise,
#                                                         size = image_size,
#                                                         target_resolution = target_resolution,
#                                                         force_overwrite = False,
#                                                         cv_fold_num = 2)

# imts = data_pros['images_test']
# gtts = data_pros['masks_test']
# name_test_subjects = data_pros['patnames_test']

# orig_data_res_x = data_pros['px_test'][:]
# orig_data_res_y = data_pros['py_test'][:]
# orig_data_res_z = data_pros['pz_test'][:]
# orig_data_siz_x = data_pros['nx_test'][:]
# orig_data_siz_y = data_pros['ny_test'][:]
# orig_data_siz_z = data_pros['nz_test'][:]

# num_test_subjects = orig_data_siz_z.shape[0] 
# ids = np.arange(num_test_subjects)

# logging.info(name_test_subjects)

# for sub_num in range(20):
#     logging.info("PROMISE subject " + str(sub_num) + ": " + str(name_test_subjects[sub_num])[2:-1] + ", res in z: " + str(orig_data_res_z[sub_num]) + ", num slices: " + str(orig_data_siz_z[sub_num]))

# ==================================================================
# Read NCI
# ==================================================================
# data_pros = data_nci.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_nci,
#                                                  preprocessing_folder = sys_config.preproc_folder_nci,
#                                                  size = image_size,
#                                                  target_resolution = target_resolution,
#                                                  force_overwrite = False,
#                                                  cv_fold_num = 1)

# imtr, gttr = [ data_pros['images_train'], data_pros['masks_train'] ]
# imvl, gtvl = [ data_pros['images_validation'], data_pros['masks_validation'] ]
# orig_data_siz_z_train = data_pros['nz_train'][:]
# orig_data_res_z_train = data_pros['pz_train'][:]
# name_train_subjects = data_pros['patnames_train']

# for sub_num in range(orig_data_siz_z_train.shape[0]):
#     logging.info("NCI subject " + str(sub_num) + ": " + str(name_train_subjects[sub_num])[2:-1] + ", res in z: " + str(orig_data_res_z_train[sub_num]) + ", num slices: " + str(orig_data_siz_z_train[sub_num]))

# ==================================================================
# Plotting SD KDEs
# ==================================================================
path_to_model = sys_config.project_root + 'log_dir/' + exp_config.expname_i2l + 'models/'
b_size = 2
alpha = 10.0
res = 0.1
x_min = -3.0
x_max = 3.0
pdf_str = 'alpha' + str(alpha) + 'xmin' + str(x_min) + 'xmax' + str(x_max) + '_res' + str(res) + '_bsize' + str(b_size)
x_values = np.arange(x_min, x_max + res, res)
sd_pdfs_filename = path_to_model + 'sd_pdfs_' + pdf_str + '_subjectwise.npy'

sd_kdes = np.load(sd_pdfs_filename)
print(sd_kdes.shape)
    
for delta in [0, 32, 96, 224, 480, 608, 672]:                
    for c in range(5):
        plt.figure(figsize=[2.5,2.5])
        for s in range(sd_kdes.shape[0]):
            plt.plot(np.arange(x_min, x_max + res, res), sd_kdes[s, c + delta, :])
        plt.savefig(path_to_model + 'sd_pdfs_' + pdf_str + '_sub' + str(s) + '_c' + str(c+delta) + '.png')
        plt.close()