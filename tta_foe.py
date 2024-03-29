# ==================================================================
# The following things happen in this file:
# ==================================================================
# 1. Get dataset dependent params (image_size, resolution, etc)
# 2. Load training data (for visualizing SD features as compared to TD features)
# 3. Load test data
# 4. Set paths and directories for the requested test ID
# 5. Extract test image for the requested test TD
# 6. Build the TF graph (normalization and segmentation networks)
# 7. Define ops for computing 1d PDFs (Gaussians / KDEs) for all channels of all layers 
# 8. Define loss functions for matching the 1D PDFs
# 9. Define optimization routine (gradient aggregation over all batches to cover the image volume)
# 10. Define summary ops
# 11. Define savers
# 12. Load SD PDFs (Gaussians / KDEs)
# 13. Load SD trained normalization and segmentation models.
# 14. TTA iterations
# 15. Visualizations:
# a. Image, normalized image and predicted labels
# b. Features of SD vs TD
# c. 1D PDFs that are being aligned
# ==================================================================

# ==================================================================
# import  
# ==================================================================
import logging
import os.path
import argparse
import numpy as np
import pickle as pk
import tensorflow as tf
import sklearn.metrics as met

import utils
import utils_vis
import utils_kde
import utils_data
import model as model
import config.params as exp_config
import config.system_paths as sys_config

# ==================================================================
# setup logging
# ==================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ==================================================================
# parse arguments
# ==================================================================
parser = argparse.ArgumentParser(prog = 'PROG')

# Training dataset and run number
parser.add_argument('--train_dataset', default = "RUNMC") # RUNMC (prostate) | CSF (cardiac) | UMC (brain white matter hyperintensities) | HCPT1 (brain subcortical tissues) | site2
parser.add_argument('--tr_run_number', type = int, default = 1) # 1 / 
parser.add_argument('--tr_cv_fold_num', type = int, default = 1) # 1 / 2
# Test dataset and subject number
parser.add_argument('--test_dataset', default = "BMC") # BMC / USZ / UCL / BIDMC / HK (prostate) | UHE / HVHD (cardiac) | UMC / NUHS (brain WMH) | CALTECH (brain tissues) | site3
parser.add_argument('--test_cv_fold_num', type = int, default = 1) # 1 / 2
parser.add_argument('--test_sub_num', type = int, default = 0) # 0 to 19

# TTA base string
parser.add_argument('--tta_string', default = "tta/")
parser.add_argument('--tta_method', default = "FoE")
# Which vars to adapt?
parser.add_argument('--TTA_VARS', default = "NORM") # BN / NORM / AdaptAx / AdaptAxAf
# Whether to use Gaussians / KDEs
parser.add_argument('--PDF_TYPE', default = "GAUSSIAN") # GAUSSIAN / KDE
# If KDEs, what smoothing parameter
parser.add_argument('--KDE_ALPHA', type = float, default = 10.0) # 10.0
# How many moments to match and how?
parser.add_argument('--LOSS_TYPE', default = "KL") # KL / EM1 / EM2
parser.add_argument('--KL_ORDER', default = "SD_vs_TD") # SD_vs_TD / TD_vs_SD
# Matching settings
parser.add_argument('--match_with_sd', type = int, default = 2) # 1 / 2 / 3 / 4

# PCA settings
parser.add_argument('--PCA_PSIZE', type = int, default = 16) # 32 / 64 / 128
parser.add_argument('--PCA_STRIDE', type = int, default = 8)
# (for UMC, where this needs to set to 2 to get enough 'fg' patches for all subjects)
parser.add_argument('--PCA_LAYER', default = 'layer_7_2') # layer_7_2 / logits / softmax
parser.add_argument('--PCA_LATENT_DIM', type = int, default = 10) # 10 / 50
parser.add_argument('--PCA_KDE_ALPHA', type = float, default = 10.0) # 0.1 / 1.0 / 10.0
parser.add_argument('--PCA_THRESHOLD', type = float, default = 0.8) # 0.8
parser.add_argument('--PCA_LAMBDA', type = float, default = 0.1) # 0.0 / 1.0 / 0.1 / 0.01 

# Batch settings
parser.add_argument('--b_size', type = int, default = 8)
# (for spine "site1", this needs to set to 2 as volumes there contain less than 8 slices)
parser.add_argument('--feature_subsampling_factor', type = int, default = 1) # 1 / 8 / 16
parser.add_argument('--features_randomized', type = int, default = 0) # 1 / 0

# Learning rate settings
parser.add_argument('--tta_learning_rate', type = float, default = 1e-4) # 0.001 / 0.0005 / 0.0001 
parser.add_argument('--tta_learning_sch', type = int, default = 0) # 0 / 1
parser.add_argument('--tta_runnum', type = int, default = 1) # 1 / 2 / 3

# weight of spectral norm loss
parser.add_argument('--lambda_spectral', type = float, default = 1.0) # 1.0 / 5.0

# whether to print debug stuff or not
parser.add_argument('--debug', type = int, default = 0) # 1 / 0

# whether to train Ax first or not
parser.add_argument('--train_Ax_first', type = int, default = 0) # 1 / 0
parser.add_argument('--instance_norm_in_Ax', type = int, default = 0) # 1 / 0

# number channels in features that are autoencoded
parser.add_argument('--num_channels_f1', type = int, default = 32) # 16 / 32
parser.add_argument('--num_channels_f2', type = int, default = 64) # 32 / 64
parser.add_argument('--num_channels_f3', type = int, default = 128) # 64 / 128

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
tta_max_steps = dataset_params[6]
tta_model_saving_freq = dataset_params[7]
tta_vis_freq = dataset_params[8]
b_size_compute_sd_pdfs = dataset_params[9]

# ================================================================
# load training data (for visualizing SD features as compared to TD features)
# ================================================================
loaded_training_data = utils_data.load_training_data(args.train_dataset,
                                                     image_size,
                                                     target_resolution)
imtr = loaded_training_data[0]
orig_data_siz_z_train = loaded_training_data[7]

# ================================================================
# load test data
# ================================================================
loaded_test_data = utils_data.load_testing_data(args.test_dataset,
                                                args.test_cv_fold_num,
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

# ================================================================
# Set paths and directories for the requested test ID
# ================================================================
sub_num = args.test_sub_num    
subject_name = str(name_test_subjects[sub_num])[2:-1]
logging.info(subject_name)

# dir where the SD mdoels have been saved
expname_i2l = 'tr' + args.train_dataset + '_cv' + str(args.tr_cv_fold_num) + '_r' + str(args.tr_run_number) + '/' + 'i2i2l/'
log_dir = sys_config.project_root + 'log_dir/' + expname_i2l

# dir where the SD PDFs have been saved
log_dir_pdfs = log_dir + 'onedpdfs/'

# dir for TTA
exp_str = exp_config.make_tta_exp_name(args, tta_method = args.tta_method) + args.test_dataset + '_' + subject_name
log_dir_tta = log_dir + exp_str
tensorboard_dir_tta = sys_config.tensorboard_root + expname_i2l + exp_str

logging.info('SD training directory: %s' %log_dir)
logging.info('Tensorboard directory TTA: %s' %tensorboard_dir_tta)

if not tf.gfile.Exists(log_dir_tta):
    tf.gfile.MakeDirs(log_dir_tta)
    tf.gfile.MakeDirs(tensorboard_dir_tta)

# ================================================================
# Run if not TTA not done before
# ================================================================
if not tf.gfile.Exists(log_dir_tta + '/models/model.ckpt-999.index'):

    # ================================================================
    # Extract test image for the requested TD
    # ================================================================
    subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])
    subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])
    test_image = imts[subject_id_start_slice:subject_id_end_slice,:,:]  
    test_image_gt = gtts[subject_id_start_slice:subject_id_end_slice,:,:]  
    test_image_gt = test_image_gt.astype(np.uint8)

    # For this test subject, determine if the pre-processing cropped out some area or padded zeros
    # If zeros were padded, determine the amount of padding (so that KDE computations can ignore this)
    nxhat = orig_data_siz_x[sub_num] * orig_data_res_x[sub_num] / target_resolution[0]
    nyhat = orig_data_siz_y[sub_num] * orig_data_res_y[sub_num] / target_resolution[1]
    if args.test_dataset == 'USZ':
        padding_y = np.maximum(0, test_image.shape[1] - nxhat.astype(np.uint16)) // 2
        padding_x = np.maximum(0, test_image.shape[2] - nyhat.astype(np.uint16)) // 2
    else:
        padding_x = np.maximum(0, test_image.shape[1] - nxhat.astype(np.uint16)) // 2
        padding_y = np.maximum(0, test_image.shape[2] - nyhat.astype(np.uint16)) // 2

    # ================================================================
    # Build the TF graph
    # ================================================================
    with tf.Graph().as_default():
        
        # ============================
        # Set random seed for reproducibility
        # ============================
        tf.random.set_random_seed(args.tta_runnum)
        np.random.seed(args.tta_runnum)
        
        # ================================================================
        # Create placeholders
        # ================================================================
        if args.PDF_TYPE == 'KDE':
            images_pl = tf.placeholder(tf.float32, shape = [args.b_size] + list(image_size) + [1], name = 'images')
        elif args.PDF_TYPE == 'GAUSSIAN':
            images_pl = tf.placeholder(tf.float32, shape = [None] + list(image_size) + [1], name = 'images')
        # setting training flag to false (relevant for batch normalization layers)
        training_pl = tf.constant(False, dtype=tf.bool)
        # placeholders for indicating amount of zero padding in the given test image
        delta_x_pl = tf.placeholder(tf.int32, shape = [], name='zero_padding_x_pl')
        delta_y_pl = tf.placeholder(tf.int32, shape = [], name='zero_padding_y_pl')

        if args.TTA_VARS in ['AdaptAxAf', 'AdaptAx']:

            # ================================================================
            # Insert a randomly initialized 1x1 'adaptor' even before the normalization module.
            # To follow the procedure used in He MedIA 2021, we will adapt this module for each test volume, and keep the normalization module fixed at the values learned on the SD.
            # ================================================================
            images_adapted = model.adapt_Ax(images_pl, exp_config, instance_norm = args.instance_norm_in_Ax)

            # ================================================================
            # Insert a normalization module in front of the segmentation network
            # the normalization module is adapted for each test image
            # ================================================================
            images_normalized, added_residual = model.normalize(images_adapted, exp_config, training_pl = training_pl)

            # ================================================================
            # Build the graph that computes predictions from the inference model
            # ================================================================    
            logits, softmax, preds = model.predict_i2l_with_adaptors(images_normalized,
                                                                     exp_config,
                                                                     training_pl = training_pl,
                                                                     nlabels = nlabels,
                                                                     return_features = False)

        else: # Directly feed the input image to the normalization module
            # ================================================================
            # Insert a normalization module in front of the segmentation network
            # the normalization module is adapted for each test image
            # ================================================================
            images_normalized, added_residual = model.normalize(images_pl, exp_config, training_pl = training_pl)

            # ================================================================
            # Build the graph that computes predictions from the inference model
            # ================================================================        
            logits, softmax, preds = model.predict_i2l(images_normalized, exp_config, training_pl = training_pl, nlabels = nlabels)
        
        # ================================================================
        # Divide the vars into different groups
        # ================================================================
        i2l_vars, normalization_vars, bn_vars, adapt_ax_vars, adapt_af_vars = model.divide_vars_into_groups(tf.global_variables())

        if args.debug == 1:
            logging.info("Ax vars")
            for v in adapt_ax_vars:
                logging.info(v.name)
                logging.info(v.shape)
            
            logging.info("Af vars")
            for v in adapt_af_vars:
                logging.info(v.name)
                logging.info(v.shape)

            logging.info("i2l vars")
            for v in i2l_vars:
                logging.info(v.name)

            logging.info("normalization vars")
            for v in normalization_vars:
                logging.info(v.name)

        # ================================================================
        # ops for initializing feature adaptors to identity
        # ================================================================
        if args.TTA_VARS in ['AdaptAx', 'AdaptAxAf']:
          
            wf1 = [v for v in tf.global_variables() if v.name == "i2l_mapper/adaptAf_A1/kernel:0"][0]
            wf2 = [v for v in tf.global_variables() if v.name == "i2l_mapper/adaptAf_A2/kernel:0"][0]
            wf3 = [v for v in tf.global_variables() if v.name == "i2l_mapper/adaptAf_A3/kernel:0"][0]
          
            if args.debug == 1:
                logging.info("Weight matrices of feature adaptors.. ")
                logging.info(wf1)
                logging.info(wf2)
                logging.info(wf3)
          
            wf1_init_pl = tf.placeholder(tf.float32, shape = [1,1,args.num_channels_f1,args.num_channels_f1], name = 'wf1_init_pl')
            wf2_init_pl = tf.placeholder(tf.float32, shape = [1,1,args.num_channels_f2,args.num_channels_f2], name = 'wf2_init_pl')
            wf3_init_pl = tf.placeholder(tf.float32, shape = [1,1,args.num_channels_f3,args.num_channels_f3], name = 'wf3_init_pl')
          
            init_wf1_op = wf1.assign(wf1_init_pl)
            init_wf2_op = wf2.assign(wf2_init_pl)
            init_wf3_op = wf3.assign(wf3_init_pl)

            init_ax_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(tf.reduce_mean(tf.square(images_adapted - images_pl)), var_list=adapt_ax_vars)

        # ================================================================
        # Set TTA vars
        # ================================================================
        if args.TTA_VARS == "BN":
            tta_vars = bn_vars
        elif args.TTA_VARS == "NORM":
            tta_vars = normalization_vars
        elif args.TTA_VARS == "AdaptAx":
            tta_vars = adapt_ax_vars
        elif args.TTA_VARS == "AdaptAxAf":
            tta_vars = adapt_ax_vars + adapt_af_vars

        # ================================================================
        # TTA loss - spectral norm of the feature adaptors
        # ================================================================
        if args.TTA_VARS in ['AdaptAx', 'AdaptAxAf']:
            loss_spectral_norm_wf1_op = model.spectral_loss(wf1)
            loss_spectral_norm_wf2_op = model.spectral_loss(wf2)
            loss_spectral_norm_wf3_op = model.spectral_loss(wf3)
            loss_spectral_norm_op = loss_spectral_norm_wf1_op + loss_spectral_norm_wf2_op + loss_spectral_norm_wf3_op

        # ================================================================
        # Gaussian matching without computing KDE
        # ================================================================
        if args.PDF_TYPE == 'GAUSSIAN':

            # placeholders for SD stats. These will be extracted after loading the SD trained model.
            sd_means_pl = tf.placeholder(tf.float32, shape = [None], name = 'sd_means')
            sd_variances_pl = tf.placeholder(tf.float32, shape = [None], name = 'sd_variances')

            # ================================================================
            # Define ops for computing parameters of Gaussian distributions at each channel of each layer
            # ================================================================
            td_means, td_variances = utils_kde.compute_1d_gaussian_parameters(args.feature_subsampling_factor,
                                                                              args.features_randomized,
                                                                              delta_x_pl,
                                                                              delta_y_pl)

            # =================================
            # Compute KL divergence between Gaussians
            # =================================
            loss_gaussian_kl_op = utils_kde.compute_kl_between_gaussian(sd_means_pl,
                                                                        sd_variances_pl,
                                                                        td_means,
                                                                        td_variances,
                                                                        order = args.KL_ORDER)

        # ================================================================
        # FULL matching WITH KDEs
        # ================================================================
        elif args.PDF_TYPE == 'KDE':

            # ==============
            # DIVIDE CHANNELS into 3 groups:
                # 1. 0 to 688 (layers 1_2 to 7_1) [range of channel means: [-0.5, 0.5], range of channel variances: [0.25, 2.5]]
                # 2. 689 to 704 (layer 7_2) [range of channel means: [-0.5, 1.5], range of channel variances: [2.5, 8.0]]
                # 3. 704 to 704 + n_labels (logits) [range of channel means: [-15, 20], range of channel variances: [15, 100]]
            # KDEs of channels in groups 2 and 3 need to be computed to an extended range of values.
            # ==============

            # ==============
            # PLACEHOLDERS
            # ==============
            # For SD PDFs (The shapes have to be hard-coded. Can't get the tile operations to work otherwise.)
            sd_kdes_g1_pl = tf.placeholder(tf.float32, shape = [688, 61], name = 'sd_kdes_g1') # shape [num_channels, num_points_along_intensity_range]
            sd_kdes_g2_pl = tf.placeholder(tf.float32, shape = [16, 101], name = 'sd_kdes_g2') # shape [num_channels, num_points_along_intensity_range]
            sd_kdes_g3_pl = tf.placeholder(tf.float32, shape = [nlabels, 101], name = 'sd_kdes_g3') # shape [num_channels, num_points_along_intensity_range]
            # For the points at which the PDFs are evaluated
            x_kde_g1_pl = tf.placeholder(tf.float32, shape = [61], name = 'x_kdes_g1') # shape [num_points_along_intensity_range]
            x_kde_g2_pl = tf.placeholder(tf.float32, shape = [101], name = 'x_kdes_g2') # shape [num_points_along_intensity_range]
            x_kde_g3_pl = tf.placeholder(tf.float32, shape = [101], name = 'x_kdes_g3') # shape [num_points_along_intensity_range]
            # For the KDE smoothing factor
            alpha_pl = tf.placeholder(tf.float32, shape = [], name = 'alpha') # shape [1]

            # ==============
            # Define ops for computing KDEs at each channel of each layer
            # ==============
            td_kdes_g1, td_kdes_g2, td_kdes_g3 = utils_kde.compute_1d_kdes(args.feature_subsampling_factor, # subsampling factor
                                                                        args.features_randomized, # whether to sample randomly or uniformly
                                                                        delta_x_pl, # zero padding information
                                                                        delta_y_pl,
                                                                        x_kde_g1_pl, # points where the KDEs have to be evaluated
                                                                        x_kde_g2_pl,
                                                                        x_kde_g3_pl,
                                                                        alpha_pl) # smoothing parameter

            # ==============
            # compute the TTA loss - add ops for all losses and select based on the argument
            # ==============
            if args.LOSS_TYPE == 'KL':
                loss_all_kl_g1_op = utils_kde.compute_kl_between_kdes(sd_kdes_g1_pl, td_kdes_g1, order = args.KL_ORDER)
                loss_all_kl_g2_op = utils_kde.compute_kl_between_kdes(sd_kdes_g2_pl, td_kdes_g2, order = args.KL_ORDER)
                loss_all_kl_g3_op = utils_kde.compute_kl_between_kdes(sd_kdes_g3_pl, td_kdes_g3, order = args.KL_ORDER)
            elif args.LOSS_TYPE =='EM1':
                loss_all_kl_g1_op = utils_kde.compute_em_between_kdes(sd_kdes_g1_pl, td_kdes_g1)
                loss_all_kl_g2_op = utils_kde.compute_em_between_kdes(sd_kdes_g2_pl, td_kdes_g2)
                loss_all_kl_g3_op = utils_kde.compute_em_between_kdes(sd_kdes_g3_pl, td_kdes_g3)
            elif args.LOSS_TYPE =='EM2':
                loss_all_kl_g1_op = utils_kde.compute_em_between_kdes(sd_kdes_g1_pl, td_kdes_g1, order = 2)
                loss_all_kl_g2_op = utils_kde.compute_em_between_kdes(sd_kdes_g2_pl, td_kdes_g2, order = 2)
                loss_all_kl_g3_op = utils_kde.compute_em_between_kdes(sd_kdes_g3_pl, td_kdes_g3, order = 2)

        # ================================================================
        # Patch extraction for PCA
        # ================================================================

        # =============
        # Combine the softmax scores into a map of foreground probabilities
        # =============
        features_fg_probs = tf.expand_dims(tf.math.reduce_max(softmax[:, :, :, 1:], axis=-1), axis=-1)

        # =============
        # Extract patches
        # =============
        patches_fg_probs = utils_kde.extract_patches(features_fg_probs,
                                                        channel = 0, 
                                                        psize = args.PCA_PSIZE,
                                                        stride = args.PCA_STRIDE)        

        # =============
        # Get indices of active patches
        # =============
        indices_active_patches = tf.squeeze(tf.where(patches_fg_probs[:, (args.PCA_PSIZE * (args.PCA_PSIZE + 1)) // 2] > args.PCA_THRESHOLD))

        # =============
        # Compute features from layer 7_2
        # =============
        features_td = tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + str(7) + '_' + str(2) + '_bn/FusedBatchNorm:0')

        # =============
        # Define PCA PLACEHOLDERS
        # =============
        # for loading saved PCA details
        pca_mean_features_pl = tf.placeholder(tf.float32, shape = [features_td.shape[-1], args.PCA_PSIZE * args.PCA_PSIZE], name = 'pca_mean')
        pca_components_pl = tf.placeholder(tf.float32, shape = [features_td.shape[-1], args.PCA_LATENT_DIM, args.PCA_PSIZE * args.PCA_PSIZE], name = 'pca_principal_comps')
        pca_variance_pl = tf.placeholder(tf.float32, shape = [features_td.shape[-1], args.PCA_LATENT_DIM], name = 'pca_variances')
        # for weight of pca kde matching loss
        lambda_pca_pl = tf.placeholder(tf.float32, shape = [], name = 'lambda_pca') # shape [1]

        # =============
        # Compute KDE for each channel of features
        # =============
        for c in range(features_td.shape[-1]):
            patches_features_td_c = utils_kde.extract_patches(features_td,
                                                                channel = c,
                                                                psize = args.PCA_PSIZE,
                                                                stride = args.PCA_STRIDE)

            # =============
            # select active patches
            # =============
            patches_features_td_c_active = tf.gather(patches_features_td_c,
                                                        indices_active_patches,
                                                        axis=0)

            # =============
            # compute PCA latent representation of these patches
            # =============
            latent_of_active_patches_td_c = utils_kde.compute_pca_latents(patches_features_td_c_active, # [num_patches, psize*psize]
                                                                            tf.gather(pca_mean_features_pl, c, axis=0), # [pca.mean_ --> [psize*psize]]
                                                                            tf.gather(pca_components_pl, c, axis=0), # [pca.components_ --> [num_components, psize*psize]]
                                                                            tf.gather(pca_variance_pl, c, axis=0)) # [pca.explained_variance_ --> [num_components]]
            # =============
            # concat to array containing the latent reps for all channels of the last layer
            # =============
            if c == 0:
                latent_of_active_patches_td = latent_of_active_patches_td_c
            else:
                latent_of_active_patches_td = tf.concat([latent_of_active_patches_td, latent_of_active_patches_td_c], -1)

        # ================================================================
        # Gaussian matching without computing KDE
        # ================================================================
        if args.PDF_TYPE == 'GAUSSIAN':
            
            # placeholders for SD latents' stats. These will be extracted after loading the SD trained model.
            sd_latents_means_pl = tf.placeholder(tf.float32, shape = [None], name = 'sd_latents_means')
            sd_latents_variances_pl = tf.placeholder(tf.float32, shape = [None], name = 'sd_latents_variances')

            # ================================================================
            # the shape of latent_of_active_patches_td is [num_samples, num_pca_latents * num_channels]
            # the shape of td_latents_means and td_latents_variances will be [num_pca_latents * num_channels]
            # ================================================================
            td_latents_means, td_latents_variances = tf.nn.moments(latent_of_active_patches_td, axes = [0]) 

            # =================================
            # Compute KL divergence between Gaussians
            # =================================
            loss_pca_kl_op = utils_kde.compute_kl_between_gaussian(sd_latents_means_pl,
                                                                   sd_latents_variances_pl,
                                                                   td_latents_means,
                                                                   td_latents_variances,
                                                                   order = args.KL_ORDER)

            # =================================
            # Total loss
            # =================================
            if args.TTA_VARS in ['AdaptAx', 'AdaptAxAf']:
                loss_op = loss_gaussian_kl_op + args.PCA_LAMBDA * loss_pca_kl_op + args.lambda_spectral * loss_spectral_norm_op
            else:
                loss_op = loss_gaussian_kl_op + args.PCA_LAMBDA * loss_pca_kl_op
                    
            # ================================================================
            # add losses to tensorboard
            # ================================================================
            tf.summary.scalar('loss/TTA', loss_op)         
            tf.summary.scalar('loss/PCA_Gaussians_KL', loss_pca_kl_op)
            tf.summary.scalar('loss/CNN_Gaussians_KL', loss_gaussian_kl_op)
            if args.TTA_VARS in ['AdaptAx', 'AdaptAxAf']:
                tf.summary.scalar('loss/spectral_loss', loss_spectral_norm_op)
            summary_during_tta = tf.summary.merge_all()

        # ================================================================
        # FULL matching WITH KDEs
        # ================================================================
        elif args.PDF_TYPE == 'KDE':

            # =============
            # Placeholders
            # =============
            # for points at which the PDFs are evaluated
            z_kde_pl = tf.placeholder(tf.float32, shape = [101], name = 'z_kdes') # shape [num_points_along_intensity_range]
            # for sd KDEs of latents
            kde_latents_sd_pl = tf.placeholder(tf.float32, shape = [160, 101], name = 'kde_latents_sd') # shape [num_channels*pca_latent_dim, num_points_along_intensity_range]

            # =============
            # compute per-dimension KDE of the latents
            # =============
            kde_latents_td = utils_kde.compute_pca_latent_kdes_tf(latent_of_active_patches_td, # [num_samples, num_pca_latents * num_channels]
                                                                  z_kde_pl,
                                                                  args.PCA_KDE_ALPHA)

            # =============
            # KL loss between SD and TD KDEs
            # =============
            if args.LOSS_TYPE == 'KL':
                loss_pca_kl_op = utils_kde.compute_kl_between_kdes(kde_latents_sd_pl, kde_latents_td, order = args.KL_ORDER)
            elif args.LOSS_TYPE == 'EM1':
                loss_pca_kl_op = utils_kde.compute_em_between_kdes(kde_latents_sd_pl, kde_latents_td)
            elif args.LOSS_TYPE == 'EM2':
                loss_pca_kl_op = utils_kde.compute_em_between_kdes(kde_latents_sd_pl, kde_latents_td, order = 2)
            
            # =================================
            # Total loss
            # =================================
            ncg1 = 688
            ncg2 = 16
            ncg3 = nlabels
            loss_all_kl_op = (ncg1 * loss_all_kl_g1_op + ncg2 * loss_all_kl_g2_op + ncg3 * loss_all_kl_g3_op) / (ncg1 + ncg2 + ncg3)
            # =================================
            # Total loss
            # =================================
            if args.TTA_VARS in ['AdaptAx', 'AdaptAxAf']:
                loss_op = loss_all_kl_op + args.PCA_LAMBDA * loss_pca_kl_op + args.lambda_spectral * loss_spectral_norm_op
            else:
                loss_op = loss_all_kl_op + args.PCA_LAMBDA * loss_pca_kl_op
                    
            # ================================================================
            # add losses to tensorboard
            # ================================================================
            tf.summary.scalar('loss/TTA', loss_op)         
            tf.summary.scalar('loss/KDE_KL_G1', loss_all_kl_g1_op)
            tf.summary.scalar('loss/KDE_KL_G2', loss_all_kl_g2_op)
            tf.summary.scalar('loss/KDE_KL_G3', loss_all_kl_g3_op)
            tf.summary.scalar('loss/KDE_KL_PCA', loss_pca_kl_op)
            tf.summary.scalar('loss/KDE_KL', loss_all_kl_op)
            if args.TTA_VARS in ['AdaptAx', 'AdaptAxAf']:
                tf.summary.scalar('loss/spectral_loss', loss_spectral_norm_op)
            summary_during_tta = tf.summary.merge_all()
        
        # ================================================================
        # Add optimization ops
        # ================================================================   
        lr_pl = tf.placeholder(tf.float32, shape = [], name = 'tta_learning_rate') # shape [1]
        # create an instance of the required optimizer
        optimizer = exp_config.optimizer_handle(learning_rate = lr_pl)    
        # initialize variable holding the accumlated gradients and create a zero-initialisation op
        accumulated_gradients = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in tta_vars]
        # accumulated gradients init op
        accumulated_gradients_zero_op = [ac.assign(tf.zeros_like(ac)) for ac in accumulated_gradients]
        # calculate gradients and define accumulation op
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            gradients = optimizer.compute_gradients(loss_op, var_list = tta_vars)
        # compute_gradients return a list of (gradient, variable) pairs.
        accumulate_gradients_op = [ac.assign_add(gg[0]) for ac, gg in zip(accumulated_gradients, gradients)]
        # define the gradient mean op
        num_accumulation_steps_pl = tf.placeholder(dtype=tf.float32, name='num_accumulation_steps')
        accumulated_gradients_mean_op = [ag.assign(tf.divide(ag, num_accumulation_steps_pl)) for ag in accumulated_gradients]
        # reassemble the gradients in the [value, var] format and do define train op
        final_gradients = [(ag, gg[1]) for ag, gg in zip(accumulated_gradients, gradients)]
        train_op = optimizer.apply_gradients(final_gradients)

        # ================================================================
        # placeholder for logging a smoothened loss
        # ================================================================                        
        loss_whole_subject_pl = tf.placeholder(tf.float32, shape = [], name = 'loss_whole_subject') # shape [1]
        loss_whole_subject_summary = tf.summary.scalar('loss/TTA_whole_subject', loss_whole_subject_pl)
        loss_ema_pl = tf.placeholder(tf.float32, shape = [], name = 'loss_ema') # shape [1]
        loss_ema_summary = tf.summary.scalar('loss/TTA_EMA', loss_ema_pl)

        # ================================================================
        # add init ops
        # ================================================================
        init_ops = tf.global_variables_initializer()
        # init_tta_ops = tf.initialize_variables(tta_vars) # set TTA vars to random values
                
        # ================================================================
        # create session
        # ================================================================
        sess = tf.Session()

        # ================================================================
        # create a summary writer
        # ================================================================
        summary_writer = tf.summary.FileWriter(tensorboard_dir_tta, sess.graph)

        # ================================================================
        # other summaries 
        # ================================================================        
        gt_dice = tf.placeholder(tf.float32, shape=[], name='gt_dice')
        gt_dice_summary = tf.summary.scalar('test_img/gt_dice', gt_dice)

        # ==============================================================================
        # define placeholder for image summaries
        # ==============================================================================    
        display_pl = tf.placeholder(tf.uint8, shape = [1, None, None, 1], name='display_pl')
        images_summary = tf.summary.image('display', display_pl)
        display_features_sd_pl = tf.placeholder(tf.uint8, shape = [1, None, None, 1], name='display_features_sd_pl')
        display_features_sd_summary = tf.summary.image('display_features_sd', display_features_sd_pl)
        display_features_td_pl = tf.placeholder(tf.uint8, shape = [1, None, None, 1], name='display_features_td_pl')
        display_features_td_summary = tf.summary.image('display_features_td', display_features_td_pl)
        display_pdfs_pl = tf.placeholder(tf.uint8, shape = [1, None, None, 1], name='display_pdfs_pl')
        pdfs_summary = tf.summary.image('PDFs', display_pdfs_pl)

        # ================================================================
        # create saver
        # ================================================================
        saver_i2l = tf.train.Saver(var_list = i2l_vars)
        saver_tta = tf.train.Saver(var_list = tta_vars, max_to_keep=1)   
        saver_tta_best = tf.train.Saver(var_list = tta_vars, max_to_keep=1) 
                
        # ================================================================
        # freeze the graph before execution
        # ================================================================
        tf.get_default_graph().finalize()

        # ================================================================
        # Run the Op to initialize the variables.
        # ================================================================
        sess.run(init_ops)
        if args.TTA_VARS in ['AdaptAx', 'AdaptAxAf']:
            sess.run(init_wf1_op, feed_dict={wf1_init_pl: np.expand_dims(np.expand_dims(np.eye(args.num_channels_f1), axis=0), axis=0)})
            sess.run(init_wf2_op, feed_dict={wf2_init_pl: np.expand_dims(np.expand_dims(np.eye(args.num_channels_f2), axis=0), axis=0)})
            sess.run(init_wf3_op, feed_dict={wf3_init_pl: np.expand_dims(np.expand_dims(np.eye(args.num_channels_f3), axis=0), axis=0)})

            if args.debug == 1:
                logging.info('Initialized feature adaptors..')   
                logging.info(wf1.eval(session=sess))
                logging.info(wf2.eval(session=sess))
                logging.info(wf3.eval(session=sess))

            if args.train_Ax_first == 1:
                logging.info('Training Ax to be the identity mapping..')
                for _ in range(100):
                    sess.run(init_ax_op, feed_dict={images_pl: np.expand_dims(test_image[np.random.randint(0, test_image.shape[0], args.b_size), :, :], axis=-1)})
                logging.info('Done.. now doing TTA ops from here..')
            
        # ================================================================
        # Restore the segmentation network parameters
        # ================================================================
        logging.info('============================================================')   
        path_to_model = sys_config.project_root + 'log_dir/' + expname_i2l + 'models/'
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_dice.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_i2l.restore(sess, checkpoint_path)

        # ================================================================
        # compute the SD Gaussian PDFs once
        # These will be passed as placeholders for computing the loss in each iteration
        # ================================================================
        if args.PDF_TYPE == 'GAUSSIAN':
            gaussians_sd = np.load(exp_config.make_sd_gaussian_names(log_dir_pdfs, args))

        # ================================================================
        # compute the SD PDFs once (extract the whole pdf instead of just the 1st and 2nd moments of the pdf)
        # These will be passed as placeholders for computing the loss in each iteration
        # ================================================================
        elif args.PDF_TYPE == 'KDE':

            kde_g1_params = [-3.0, 3.0, 0.1]
            kde_g2_params = [-5.0, 5.0, 0.1]
            kde_g3_params = [0.0, 1.0, 0.01]
            kde_z_params = [-5.0, 5.0, 0.1]

            # make array of points where the KDEs have to be evaluated
            x_values_g1 = np.arange(kde_g1_params[0], kde_g1_params[1] + kde_g1_params[2], kde_g1_params[2])
            x_values_g2 = np.arange(kde_g2_params[0], kde_g2_params[1] + kde_g2_params[2], kde_g2_params[2])
            x_values_g3 = np.arange(kde_g3_params[0], kde_g3_params[1] + kde_g3_params[2], kde_g3_params[2])
            z_values = np.arange(kde_z_params[0], kde_z_params[1] + kde_z_params[2], kde_z_params[2])

            kdes_sd_g1 = np.load(log_dir_pdfs + exp_config.make_sd_kde_name(b_size_compute_sd_pdfs, args.KDE_ALPHA, kde_g1_params) + '_g1.npy')
            kdes_sd_g2 = np.load(log_dir_pdfs + exp_config.make_sd_kde_name(b_size_compute_sd_pdfs, args.KDE_ALPHA, kde_g2_params) + '_g2.npy')
            kdes_sd_g3 = np.load(log_dir_pdfs + exp_config.make_sd_kde_name(b_size_compute_sd_pdfs, args.KDE_ALPHA, kde_g3_params) + '_g3.npy')

        # =============================
        # LOAD PCA RESULTS
        # =============================            
        pca_dir = log_dir_pdfs + exp_config.make_pca_dir_name(args)
        num_pca_channels = 16
        pca_means = []
        pca_pcs = []
        pca_vars = []

        for c in range(num_pca_channels):

            # load pca components, mean, variances
            pca_c = pk.load(open(pca_dir + 'c' + str(c) + '.pkl', 'rb'))
            pca_means.append(pca_c.mean_)
            pca_pcs.append(pca_c.components_)
            pca_vars.append(pca_c.explained_variance_)

            # load sd kdes / gaussians
            pca_latent_sd_kdes_c = np.load(pca_dir + 'c' + str(c) + '.npy') 
            # if KDEs: [num_sd_subjects, num_latents, num_z_values]
            # if GAUSSIANS: [num_sd_subjects, num_latents, 2]
            if c == 0:
                pca_latent_sd_kdes = pca_latent_sd_kdes_c
            else:
                pca_latent_sd_kdes = np.concatenate((pca_latent_sd_kdes, pca_latent_sd_kdes_c), axis=1)

        pca_means = np.array(pca_means)
        pca_pcs = np.array(pca_pcs)
        pca_vars = np.array(pca_vars)

        # ===========================
        # get dice wrt ground truth before any TTA updates have been done
        # ===========================
        b_size = args.b_size
        label_predicted = []
        for b_i in range(0, test_image.shape[0], b_size):
            if b_i + b_size < test_image.shape[0]:
                batch = np.expand_dims(test_image[b_i:b_i+b_size, ...], axis=-1)
            else:
                # pad zeros to have complete batches
                extra_zeros_needed = b_i + b_size - test_image.shape[0]
                batch = np.expand_dims(np.concatenate((test_image[b_i:, ...], np.zeros((extra_zeros_needed, test_image.shape[1], test_image.shape[2]))), axis=0), axis=-1)
            label_predicted.append(sess.run(preds, feed_dict={images_pl: batch}))
        label_predicted = np.squeeze(np.array(label_predicted)).astype(float)  
        if b_size > 1 and test_image.shape[0] > b_size:
            label_predicted = np.reshape(label_predicted, (label_predicted.shape[0]*label_predicted.shape[1], label_predicted.shape[2], label_predicted.shape[3]))
        label_predicted = label_predicted[:test_image.shape[0], ...]
        if args.test_dataset in ['UCL', 'HK', 'BIDMC']:
            label_predicted[label_predicted!=0.0] = 1.0
        dice_wrt_gt = met.f1_score(test_image_gt.flatten(), label_predicted.flatten(), average=None) 
        summary_writer.add_summary(sess.run(gt_dice_summary, feed_dict={gt_dice: np.mean(dice_wrt_gt[1:])}), 0)

        # ===================================
        # TTA / SFDA iterations
        # ===================================
        step = 0
        best_loss = 100000.0

        while (step < tta_max_steps):
            
            logging.info("TTA step: " + str(step+1))
            
            # =============================
            # run accumulated_gradients_zero_op (no need to provide values for any placeholders)
            # =============================
            sess.run(accumulated_gradients_zero_op)
            num_accumulation_steps = 0
            loss_this_step = 0.0

            # =============================
            # Learning rate schedule
            # =============================
            if args.tta_learning_sch == 1:
                if step < tta_max_steps // 2:
                    tta_learning_rate = args.tta_learning_rate
                else:
                    tta_learning_rate = args.tta_learning_rate / 10.0
            elif args.tta_learning_sch == 0:
                tta_learning_rate = args.tta_learning_rate

            # =============================
            # SD PDF / Gaussian to match with
            # =============================
            # Match with mean PDF over SD subjects
            if args.match_with_sd == 1: 
                if args.PDF_TYPE == 'KDE':
                    sd_kdes_g1_this_step = np.mean(kdes_sd_g1, axis = 0)
                    sd_kdes_g2_this_step = np.mean(kdes_sd_g2, axis = 0)
                    sd_kdes_g3_this_step = np.mean(kdes_sd_g3, axis = 0)
                    sd_kde_latents_this_step = np.mean(pca_latent_sd_kdes, axis = 0)
                elif args.PDF_TYPE == 'GAUSSIAN':
                    sd_gaussian_this_step = np.mean(gaussians_sd, axis=0)
                    sd_gaussian_latents_this_step = np.mean(pca_latent_sd_kdes, axis=0)

            # Select a different SD subject for each TTA iteration
            elif args.match_with_sd == 2: 
                if args.PDF_TYPE == 'KDE':
                    sub_id = np.random.randint(kdes_sd_g1.shape[0])  
                    sd_kdes_g1_this_step = kdes_sd_g1[sub_id, :, :]
                    sd_kdes_g2_this_step = kdes_sd_g2[sub_id, :, :]
                    sd_kdes_g3_this_step = kdes_sd_g3[sub_id, :, :]
                    sd_kde_latents_this_step = pca_latent_sd_kdes[sub_id, :, :]
                elif args.PDF_TYPE == 'GAUSSIAN':
                    sub_id = np.random.randint(gaussians_sd.shape[0])
                    sd_gaussian_this_step = gaussians_sd[sub_id, :, :]
                    sd_gaussian_latents_this_step = pca_latent_sd_kdes[sub_id, :, :]
                            
            # =============================
            # Adaptation iterations within this epoch
            # =============================
            b_size = args.b_size
            for b_i in range(0, test_image.shape[0], b_size):

                if args.PDF_TYPE == 'KDE':

                    feed_dict={images_pl: np.expand_dims(test_image[np.random.randint(0, test_image.shape[0], b_size), :, :], axis=-1),
                               sd_kdes_g1_pl: sd_kdes_g1_this_step, 
                               sd_kdes_g2_pl: sd_kdes_g2_this_step, 
                               sd_kdes_g3_pl: sd_kdes_g3_this_step, 
                               x_kde_g1_pl: x_values_g1, 
                               x_kde_g2_pl: x_values_g2, 
                               x_kde_g3_pl: x_values_g3, 
                               alpha_pl: args.KDE_ALPHA,
                               lr_pl: tta_learning_rate,
                               delta_x_pl: padding_x,
                               delta_y_pl: padding_y,
                               pca_mean_features_pl: pca_means,
                               pca_components_pl: pca_pcs,
                               pca_variance_pl: pca_vars,
                               z_kde_pl: z_values,
                               kde_latents_sd_pl: sd_kde_latents_this_step,
                               lambda_pca_pl: args.PCA_LAMBDA}

                elif args.PDF_TYPE == 'GAUSSIAN':

                    feed_dict={images_pl: np.expand_dims(test_image[np.random.randint(0, test_image.shape[0], b_size), :, :], axis=-1),
                               sd_means_pl: sd_gaussian_this_step[:,0], 
                               sd_variances_pl: sd_gaussian_this_step[:,1],
                               sd_latents_means_pl: sd_gaussian_latents_this_step[:,0], 
                               sd_latents_variances_pl: sd_gaussian_latents_this_step[:,1],
                               lr_pl: tta_learning_rate,
                               delta_x_pl: padding_x,
                               delta_y_pl: padding_y,
                               pca_mean_features_pl: pca_means,
                               pca_components_pl: pca_pcs,
                               pca_variance_pl: pca_vars,
                               lambda_pca_pl: args.PCA_LAMBDA}

                # run the accumulate gradients op 
                sess.run(accumulate_gradients_op, feed_dict=feed_dict)
                loss_this_step = loss_this_step + sess.run(loss_op, feed_dict = feed_dict)
                num_accumulation_steps = num_accumulation_steps + 1

            loss_this_step = loss_this_step / num_accumulation_steps # average loss (over all slices of the image volume) in this step

            # ===========================
            # save best model so far (based on an exponential moving average of the TTA loss)
            # ===========================
            momentum = 0.95
            if step == 0:
                loss_ema = loss_this_step
            else:
                loss_ema = momentum * loss_ema + (1 - momentum) * loss_this_step
            summary_writer.add_summary(sess.run(loss_ema_summary, feed_dict={loss_ema_pl: loss_ema}), step)

            if best_loss > loss_ema:
                best_loss = loss_ema
                best_file = os.path.join(log_dir_tta, 'models/best_loss.ckpt')
                saver_tta_best.save(sess, best_file, global_step=step)
                logging.info('Found new best score (%f) at step %d -  Saving model.' % (best_loss, step))

            # ===========================
            # run accumulated_gradients_mean_op, with a value for the placeholder num_accumulation_steps_pl; followed by the train_op with applies the gradients
            # ===========================
            sess.run(accumulated_gradients_mean_op, feed_dict = {num_accumulation_steps_pl: num_accumulation_steps})
            # run the train_op.
            # this also requires input output placeholders, as compute_gradients will be called again..
            # But the returned gradient values will be replaced by the mean gradients.
            sess.run(train_op, feed_dict = feed_dict)

            # ===========================
            # Periodically save models
            # ===========================
            if (step+1) % tta_model_saving_freq == 0:
                saver_tta.save(sess, os.path.join(log_dir_tta, 'models/model.ckpt'), global_step=step)

            # ===========================
            # get dice wrt ground truth
            # ===========================
            label_predicted = []
            image_normalized = []

            for b_i in range(0, test_image.shape[0], b_size):
                if b_i + b_size < test_image.shape[0]:
                    batch = np.expand_dims(test_image[b_i:b_i+b_size, ...], axis=-1)
                else:
                    # pad zeros to have complete batches
                    extra_zeros_needed = b_i + b_size - test_image.shape[0]
                    batch = np.expand_dims(np.concatenate((test_image[b_i:, ...],
                                           np.zeros((extra_zeros_needed,
                                                     test_image.shape[1],
                                                     test_image.shape[2]))), axis=0), axis=-1)
                label_predicted.append(sess.run(preds, feed_dict={images_pl: batch}))
                image_normalized.append(sess.run(images_normalized, feed_dict={images_pl: batch}))

            label_predicted = np.squeeze(np.array(label_predicted)).astype(float)  
            image_normalized = np.squeeze(np.array(image_normalized)).astype(float)  

            if b_size > 1 and test_image.shape[0] > b_size:
                label_predicted = np.reshape(label_predicted,
                                            (label_predicted.shape[0]*label_predicted.shape[1],
                                            label_predicted.shape[2],
                                            label_predicted.shape[3]))
                
                image_normalized = np.reshape(image_normalized,
                                             (image_normalized.shape[0]*image_normalized.shape[1],
                                             image_normalized.shape[2],
                                             image_normalized.shape[3]))
                
            label_predicted = label_predicted[:test_image.shape[0], ...]
            image_normalized = image_normalized[:test_image.shape[0], ...]

            if args.test_dataset in ['UCL', 'HK', 'BIDMC']:
                label_predicted[label_predicted!=0.0] = 1.0
            dice_wrt_gt = met.f1_score(test_image_gt.flatten(), label_predicted.flatten(), average=None) 
            summary_writer.add_summary(sess.run(gt_dice_summary, feed_dict={gt_dice: np.mean(dice_wrt_gt[1:])}), step)

            # ===========================
            # Update the events file
            # ===========================
            summary_str = sess.run(summary_during_tta, feed_dict = feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            # ===========================   
            # visualize 
            # ===========================
            if step % tta_vis_freq == 0:
                utils_vis.write_image_summaries(step,
                                                summary_writer,
                                                sess,
                                                images_summary,
                                                display_pl,
                                                test_image,
                                                image_normalized,
                                                label_predicted,
                                                test_image_gt,
                                                test_image_gt,
                                                padding_x,
                                                padding_y)

                # ===========================
                # Display 7_2 layer features
                # ===========================
                display_features = 1
                if display_features == 1:

                    # Get Test image features
                    tmp = test_image.shape[0] // 2 - b_size//2
                    features_for_display_td = sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv7_2_bn/FusedBatchNorm:0'),
                                                    feed_dict={images_pl: np.expand_dims(test_image[tmp:tmp+b_size, ...], axis=-1)})

                    # Get SD image features
                    while True:
                        train_sub_num = np.random.randint(orig_data_siz_z_train.shape[0])
                        if args.train_dataset == 'HCPT1': # circumventing a bug in the way orig_data_siz_z_train is written for HCP images
                            sd_image = imtr[train_sub_num*image_depth_tr : (train_sub_num+1)*image_depth_tr,:,:]
                        else:
                            sd_image = imtr[np.sum(orig_data_siz_z_train[:train_sub_num]) : np.sum(orig_data_siz_z_train[:train_sub_num+1]),:,:]
                        # move forward once you have an image that is at least as large as the batch size
                        if (sd_image.shape[0] >= b_size):
                            break
                    
                    # Select a batch from the center of the SD image
                    tmp = sd_image.shape[0] // 2 - b_size//2
                    features_for_display_sd = sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv7_2_bn/FusedBatchNorm:0'),
                                                       feed_dict={images_pl: np.expand_dims(sd_image[tmp:tmp+b_size, ...], axis=-1)})
                    
                    # Display
                    utils_vis.write_feature_summaries(step,
                                                      summary_writer,
                                                      sess,
                                                      display_features_sd_summary,
                                                      display_features_sd_pl,
                                                      features_for_display_sd,
                                                      display_features_td_summary,
                                                      display_features_td_pl,
                                                      features_for_display_td)

                # ===========================
                # Visualize KDE / Gaussian alignment
                # ===========================
                if args.PDF_TYPE == 'GAUSSIAN':

                    b_size = args.b_size
                    num_batches = 0
                    for b_i in range(0, test_image.shape[0], b_size):
                        if b_i + b_size < test_image.shape[0]: # ignoring the rest of the image (that doesn't fit the last batch) for now.                    
                            b_cnn_mu, b_cnn_var = sess.run([td_means, td_variances], feed_dict={images_pl: np.expand_dims(test_image[b_i:b_i+b_size, ...], axis=-1),
                                                                                                delta_x_pl: padding_x,
                                                                                                delta_y_pl: padding_y})
                            b_pca_mu, b_pca_var = sess.run([td_latents_means, td_latents_variances], feed_dict={images_pl: np.expand_dims(test_image[b_i:b_i+b_size, ...], axis=-1),
                                                                                                                delta_x_pl: padding_x,
                                                                                                                delta_y_pl: padding_y,
                                                                                                                pca_mean_features_pl: pca_means,
                                                                                                                pca_components_pl: pca_pcs,
                                                                                                                pca_variance_pl: pca_vars})
                            if b_i == 0:
                                test_cnn_mu = b_cnn_mu
                                test_cnn_var = b_cnn_var
                                test_pca_mu = b_pca_mu
                                test_pca_var = b_pca_var
                            else:
                                test_cnn_mu = test_cnn_mu + b_cnn_mu
                                test_cnn_var = test_cnn_var + b_cnn_var
                                test_pca_mu = test_pca_mu + b_pca_mu
                                test_pca_var = test_pca_var + b_pca_var
                            num_batches = num_batches + 1
                    test_cnn_mu = test_cnn_mu / num_batches
                    test_cnn_var = test_cnn_var / num_batches
                    test_pca_mu = test_pca_mu / num_batches
                    test_pca_var = test_pca_var / num_batches

                    utils_vis.write_gaussians(step,
                                              summary_writer,
                                              sess,
                                              pdfs_summary,
                                              display_pdfs_pl,
                                              np.mean(gaussians_sd, axis=0)[:,0],
                                              np.mean(gaussians_sd, axis=0)[:,1],
                                              test_cnn_mu,
                                              test_cnn_var,
                                              log_dir_tta,
                                              nlabels)

                    # save individual figures     
                    track_tta_evolution = False
                    if track_tta_evolution == True:
                        if not tf.gfile.Exists(log_dir_tta + '/TTA_Evolution_Figs/'):
                            tf.gfile.MakeDirs(log_dir_tta + '/TTA_Evolution_Figs/')
                        # PCA PDF matching to be added
                        utils_vis.save_indivudual_figures_tta_foe(step,
                                                                  test_image,
                                                                  image_normalized,
                                                                  label_predicted,
                                                                  test_image_gt,
                                                                  gaussians_sd[:,:,0], # means for all subjects (axis 0) for all channels (axis 1)
                                                                  gaussians_sd[:,:,1], # vars for all subjects (axis 0) for all channels (axis 1)
                                                                  test_cnn_mu, # means for this test subject for all channels
                                                                  test_cnn_var, # vars for this test subject for all channels
                                                                  pca_latent_sd_kdes[:,:,0], # means for all subjects (axis 0) for all pcs (axis 1)
                                                                  pca_latent_sd_kdes[:,:,1], # vars for all subjects (axis 0) for all pcs (axis 1)
                                                                  test_pca_mu, # means for this test subject for all pcs
                                                                  test_pca_var, # vars for this test subject for all pcs
                                                                  log_dir_tta + '/TTA_Evolution_Figs/')

                elif args.PDF_TYPE == 'KDE':
                    b_size = args.b_size
                    num_batches = 0

                    # initial batches in brain datasets are only background. This leads to no 'active' patches, which creates problems in pca computations..
                    if args.train_dataset == 'HCPT1':
                        b_i_start = 50
                    else:
                        b_i_start = 0
                    
                    for b_i in range(b_i_start, test_image.shape[0]-b_i_start, b_size):
                        
                        if b_i + b_size < test_image.shape[0]: # ignoring the rest of the image (that doesn't fit the last batch) for now.

                            batch_tmp = np.expand_dims(test_image[b_i:b_i+b_size, ...], axis=-1)
                            
                            kdes_g1_this_batch = sess.run(td_kdes_g1, feed_dict={images_pl: batch_tmp,
                                                                                 x_kde_g1_pl: x_values_g1,
                                                                                 alpha_pl: args.KDE_ALPHA,
                                                                                 delta_x_pl: padding_x,
                                                                                 delta_y_pl: padding_y})
                            
                            kdes_g2_this_batch = sess.run(td_kdes_g2, feed_dict={images_pl: batch_tmp,
                                                                                 x_kde_g2_pl: x_values_g2,
                                                                                 alpha_pl: args.KDE_ALPHA,
                                                                                 delta_x_pl: padding_x,
                                                                                 delta_y_pl: padding_y})
                            
                            kdes_g3_this_batch = sess.run(td_kdes_g3, feed_dict={images_pl: batch_tmp,
                                                                                 x_kde_g3_pl: x_values_g3,
                                                                                 alpha_pl: args.KDE_ALPHA,
                                                                                 delta_x_pl: padding_x,
                                                                                 delta_y_pl: padding_y})

                            kdes_latents_this_batch = sess.run(kde_latents_td, feed_dict={images_pl: batch_tmp,
                                                                                          z_kde_pl: z_values,
                                                                                          pca_mean_features_pl: pca_means,
                                                                                          pca_components_pl: pca_pcs,
                                                                                          pca_variance_pl: pca_vars})

                            if b_i == b_i_start:
                                kdes_td_g1_this_step = kdes_g1_this_batch
                                kdes_td_g2_this_step = kdes_g2_this_batch
                                kdes_td_g3_this_step = kdes_g3_this_batch
                                kdes_td_pca_this_step = kdes_latents_this_batch
                            else:
                                kdes_td_g1_this_step = kdes_td_g1_this_step + kdes_g1_this_batch
                                kdes_td_g2_this_step = kdes_td_g2_this_step + kdes_g2_this_batch
                                kdes_td_g3_this_step = kdes_td_g3_this_step + kdes_g3_this_batch
                                kdes_td_pca_this_step = kdes_td_pca_this_step + kdes_latents_this_batch
                            num_batches = num_batches + 1

                    kdes_td_g1_this_step = kdes_td_g1_this_step / num_batches
                    kdes_td_g2_this_step = kdes_td_g2_this_step / num_batches
                    kdes_td_g3_this_step = kdes_td_g3_this_step / num_batches
                    kdes_td_pca_this_step = kdes_td_pca_this_step / num_batches

                    utils_vis.write_pdfs(step,
                                        summary_writer,
                                        sess,
                                        pdfs_summary,
                                        display_pdfs_pl,
                                        np.mean(kdes_sd_g1, axis = 0), np.std(kdes_sd_g1, axis = 0), kdes_td_g1_this_step, x_values_g1,
                                        np.mean(kdes_sd_g2, axis = 0), np.std(kdes_sd_g2, axis = 0), kdes_td_g2_this_step, x_values_g2,
                                        np.mean(kdes_sd_g3, axis = 0), np.std(kdes_sd_g3, axis = 0), kdes_td_g3_this_step, x_values_g3,
                                        np.mean(pca_latent_sd_kdes, axis = 0), np.std(pca_latent_sd_kdes, axis = 0), kdes_td_pca_this_step, z_values,
                                        log_dir_tta)

            step = step + 1

        # ================================================================
        # close session
        # ================================================================
        sess.close()