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
import config.system_paths as sys_config
import config.params as exp_config
import argparse
from scipy import interpolate
import tensorflow_probability as tfp
import utils_kde

import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 

# ==================================================================
# setup logging
# ==================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ==================================================================
# parse arguments
# ==================================================================
parser = argparse.ArgumentParser(prog = 'PROG')
# Training dataset and run number
parser.add_argument('--train_dataset', default = "HCPT1") # NCI / HCPT1
parser.add_argument('--tr_run_number', type = int, default = 1) # 1 / 
# Test dataset 
parser.add_argument('--test_dataset', default = "HCPT2") # PROMISE / USZ / CALTECH / STANFORD / HCPT2
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
tta_max_steps = dataset_params[6]
tta_model_saving_freq = dataset_params[7]
tta_vis_freq = dataset_params[8]

# ==================================================================
# Read Training Data
# ==================================================================
# imtr, gttr, orig_data_siz_z_train, num_train_subjects = utils.load_training_data(args.train_dataset,
#                                                                                  image_size,
#                                                                                  target_resolution)

# for sub_num in range(20):
#     logging.info("Subject " + str(sub_num) + ": num slices: " + str(orig_data_siz_z_train[sub_num]))
#     subject_id_start_slice = np.sum(orig_data_siz_z_train[:sub_num])
#     subject_id_end_slice = np.sum(orig_data_siz_z_train[:sub_num+1])
#     image = imtr[subject_id_start_slice:subject_id_end_slice,:,:]  
#     logging.info("Train image shape " + str(image.shape))

# # ==================================================================
# # Read Test Data
# # ==================================================================
# loaded_test_data = utils.load_testing_data(args.test_dataset,
#                                            image_size,
#                                            target_resolution,
#                                            image_depth_ts)

# imts = loaded_test_data[0]
# orig_data_res_x = loaded_test_data[2]
# orig_data_res_y = loaded_test_data[3]
# orig_data_res_z = loaded_test_data[4]
# orig_data_siz_x = loaded_test_data[5]
# orig_data_siz_y = loaded_test_data[6]
# orig_data_siz_z = loaded_test_data[7]
# name_test_subjects = loaded_test_data[8]
# num_test_subjects = loaded_test_data[9]
# ids = loaded_test_data[10]

# logging.info(name_test_subjects)
# for sub_num in range(20):
#     logging.info("Subject " + str(sub_num) + ": " + str(name_test_subjects[sub_num])[2:-1] + ", res in z: " + str(orig_data_res_z[sub_num]) + ", num slices: " + str(orig_data_siz_z[sub_num]))
#     subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])
#     subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])
#     image = imts[subject_id_start_slice:subject_id_end_slice,:,:]  
#     logging.info("Test image shape " + str(image.shape))

# ==================================================================
# Plotting SD KDEs
# ==================================================================
# expname_i2l = 'tr' + args.train_dataset + '_r' + str(args.tr_run_number) + '/' + 'i2i2l/'
# path_to_model = sys_config.project_root + 'log_dir/' + expname_i2l + 'models/'
# b_size = 2
# alpha = 10.0
# res = 0.1
# x_min = -3.0
# x_max = 3.0
# pdf_str = 'alpha' + str(alpha) + 'xmin' + str(x_min) + 'xmax' + str(x_max) + '_res' + str(res) + '_bsize' + str(b_size)
# x_values = np.arange(x_min, x_max + res, res)
# sd_pdfs_filename = path_to_model + 'sd_pdfs_' + pdf_str + '_subjectwise.npy'
# sd_pdfs = np.load(sd_pdfs_filename)
    
# # for delta in [0, 32, 96, 224, 480, 608, 672]:                
# #     for c in range(5):
# #         plt.figure(figsize=[2.5,2.5])
# #         for s in range(sd_pdfs.shape[0]):
# #             plt.plot(np.arange(x_min, x_max + res, res), sd_pdfs[s, c + delta, :])
# #         plt.savefig(path_to_model + 'sd_pdfs_' + pdf_str + '_sub' + str(s) + '_c' + str(c+delta) + '.png')
# #         plt.close()

# ==================================================================
# Inverse Transform Sampling
# ==================================================================
# s=0
# c=0
# delta=0
# x_values = np.arange(x_min, x_max + res, res)
# sd_pdfs_one_subject = sd_pdfs[s,:,:]
# sd_cdfs_one_subject = np.cumsum(sd_pdfs_one_subject, 1)
# sd_cdfs_one_subject = sd_cdfs_one_subject / np.tile(np.max(sd_cdfs_one_subject, 1), (sd_pdfs_one_subject.shape[1], 1)).T
# # interpolate each channel separately
# for delta in [0, 672]:                
#     for c in range(3):
#         sd_cdfs_one_subject_one_channel = sd_cdfs_one_subject[c+delta,:]
#         sd_cdfs_one_subject_one_channel_inverse = interpolate.interp1d(sd_cdfs_one_subject_one_channel, x_values) # this is a function
#         # sample uniform from 0 to 1 and compute inverse CDF at these values
#         idx_uniform = np.random.uniform(low=0.0, high=1.0, size=50)
#         samples = sd_cdfs_one_subject_one_channel_inverse(idx_uniform)
#         plt.figure(figsize=[5,5])
#         plt.plot(x_values, sd_pdfs_one_subject[c+delta,:])
#         plt.plot(x_values, sd_cdfs_one_subject_one_channel)
#         plt.scatter(samples, np.zeros_like(samples),s=2)
#         plt.savefig(path_to_model + 'sd_pdf_cdfs_' + pdf_str + '_sub' + str(s) + '_c' + str(c+delta) + '.png')
#         plt.close()

# # ================================================================
# # Sample using np.random.choice
# # ================================================================
# for delta in [0, 672]:                
#     for c in range(3):
#         sd_pdfs_one_subject_one_channel = sd_pdfs_one_subject[c+delta,:]
#         sd_pdfs_one_subject_one_channel = sd_pdfs_one_subject_one_channel / np.sum(sd_pdfs_one_subject_one_channel)
#         samples = np.random.choice(x_values, 50, p=sd_pdfs_one_subject_one_channel)
#         plt.figure(figsize=[5,5])
#         plt.plot(x_values, sd_pdfs_one_subject[c+delta,:])
#         plt.scatter(samples, np.zeros_like(samples),s=2)
#         plt.savefig(path_to_model + 'sd_pdf_samples_' + pdf_str + '_sub' + str(s) + '_c' + str(c+delta) + '.png')
#         plt.close()

# sample_indices = utils_kde.sample_sd_points(sd_pdfs_one_subject, 50, x_values)
# for delta in [0, 672]:                
#     for c in range(3):
#         sd_pdfs_one_subject_one_channel = sd_pdfs_one_subject[c+delta,:]
#         sd_pdfs_one_subject_one_channel = sd_pdfs_one_subject_one_channel / np.sum(sd_pdfs_one_subject_one_channel)
        
#         plt.figure(figsize=[5,5])
#         plt.plot(x_values, sd_pdfs_one_subject[c+delta,:])
#         plt.scatter(x_values[sample_indices[c+delta,:,1]], np.zeros_like(sample_indices[c+delta,:,1]),s=2)
#         plt.savefig(path_to_model + 'sd_pdf_samples_v2_' + pdf_str + '_sub' + str(s) + '_c' + str(c+delta) + '.png')
#         plt.close()


# ================================================================
# Lebesgue integral in TF
# ================================================================
# with tf.Graph().as_default():

#     sd_pdf_pl = tf.placeholder(tf.float32, shape = [704, 61], name = 'sd_pdfs') # shape [num_channels, num_points_along_intensity_range]
#     # placeholder for the points at which the PDFs are evaluated
#     x_pdf_pl = tf.placeholder(tf.float32, shape = [61], name = 'x_pdfs') # shape [num_points_along_intensity_range]
#     # placeholder for x_values sampled from sd_pdfs (will be used to compute Lebesgue integral in KL divergence)
#     num_pts_lebesgue = 50
#     x_indices_lebesgue_pl = tf.placeholder(tf.int32, shape = [704, num_pts_lebesgue, 2], name = 'x_indices_lebesgue') # shape [num_channels, num_indices]

#     print(sd_pdf_pl)
#     print(tf.gather_nd(sd_pdf_pl, x_indices_lebesgue_pl))

# ================================================================
# print op names
# ================================================================
with tf.Graph().as_default():
    
    # ================================================================
    # create placeholders
    # ================================================================
    images_pl = tf.placeholder(tf.float32, shape = [16] + list(image_size) + [1], name = 'images')
    training_pl = tf.constant(False, dtype=tf.bool)
    # ================================================================
    # insert a normalization module in front of the segmentation network
    # the normalization module is trained for each test image
    # ================================================================
    images_normalized, added_residual = model.normalize(images_pl, exp_config, training_pl = training_pl)
    # ================================================================
    # build the graph that computes predictions from the inference model
    # ================================================================
    logits, softmax, preds = model.predict_i2l(images_normalized, exp_config, training_pl = training_pl, nlabels = nlabels)

    for conv_block in [1,2,3,4,5,6,7]:
        for conv_sub_block in [1,2]:
            conv_string = str(conv_block) + '_' + str(conv_sub_block)
            features_td = tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '_bn/FusedBatchNorm:0')
            print(features_td)

    features_td = tf.get_default_graph().get_tensor_by_name('i2l_mapper/pred/Conv2D:0')
    print(features_td)
