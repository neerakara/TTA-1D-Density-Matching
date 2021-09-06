# ==================================================================
# import 
# ==================================================================
import logging
import os.path
import tensorflow as tf
import numpy as np
import utils
import utils_data
import utils_kde
import model as model
import config.system_paths as sys_config
import config.params as exp_config
import argparse

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
parser.add_argument('--tr_cv_fold_num', type = int, default = 1) # 1 / 2
# Batch settings
parser.add_argument('--feature_subsampling_factor', type = int, default = 16) # 1 / 8 / 16
parser.add_argument('--features_randomized', type = int, default = 1) # 1 / 0
# KDE hyperparameters
parser.add_argument('--KDE_ALPHA', type = float, default = 10.0) # 10.0 / 100.0 / 1000.0
# parse arguments
args = parser.parse_args()

# ================================================================
# set dataset dependent hyperparameters
# ================================================================
dataset_params = exp_config.get_dataset_dependent_params(args.train_dataset) 
image_size = dataset_params[0]
nlabels = dataset_params[1]
target_resolution = dataset_params[2]
image_depth_tr = dataset_params[3]
b_size_compute_sd_pdfs = dataset_params[9]

# ================================================================
# load training data for computing SD PDFs
# ================================================================
loaded_training_data = utils_data.load_training_data(args.train_dataset,
                                                     image_size,
                                                     target_resolution)
imtr = loaded_training_data[0]
gttr = loaded_training_data[1]
orig_data_res_x = loaded_training_data[2]
orig_data_res_y = loaded_training_data[3]
orig_data_res_z = loaded_training_data[4]
orig_data_siz_x = loaded_training_data[5]
orig_data_siz_y = loaded_training_data[6]
orig_data_siz_z = loaded_training_data[7]

# ================================================================
# Setup directories for this run
# ================================================================
if args.train_dataset == 'UMC':
    expname_i2l = 'tr' + args.train_dataset + '_cv' + str(args.tr_cv_fold_num) + '_r' + str(args.tr_run_number) + '/' + 'i2i2l/'
else:
    expname_i2l = 'tr' + args.train_dataset + '_r' + str(args.tr_run_number) + '/' + 'i2i2l/'
log_dir = sys_config.project_root + 'log_dir/' + expname_i2l
logging.info('SD training directory: %s' %log_dir)
log_dir_pdfs = log_dir + 'onedpdfs/'
if not tf.gfile.Exists(log_dir_pdfs):
    tf.gfile.MakeDirs(log_dir_pdfs)

# ================================================================
# build the TF graph
# ================================================================
with tf.Graph().as_default():
    
    # ============================
    # set random seed for reproducibility
    # ============================
    tf.random.set_random_seed(args.tr_run_number)
    np.random.seed(args.tr_run_number)
    
    # ================================================================
    # create placeholders
    # ================================================================
    images_pl = tf.placeholder(tf.float32, shape = [b_size_compute_sd_pdfs] + list(image_size) + [1], name = 'images')
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
    # ================================================================
    # divide the vars into segmentation network and normalization network
    # ================================================================
    i2l_vars = []
    normalization_vars = []
    bn_vars = []
    for v in tf.global_variables():
        var_name = v.name        
        i2l_vars.append(v)

    # ================================================================
    # placeholders for indicating amount of zero padding in this test image
    # ================================================================
    delta_x_pl = tf.placeholder(tf.int32, shape = [], name='zero_padding_x_pl')
    delta_y_pl = tf.placeholder(tf.int32, shape = [], name='zero_padding_y_pl')

    # ================================================================
    # Define ops for computing the KDEs at each channel of each layer
    # ================================================================

    # ==========================
    # DIVIDE CHANNELS into 3 groups:
        # 1. 0 to 688 (layers 1_2 to 7_1) [range of channel means: [-0.5, 0.5], range of channel variances: [0.25, 2.5]]
        # 2. 689 to 704 (layer 7_2) [range of channel means: [-0.5, 1.5], range of channel variances: [2.5, 8.0]]
        # 3. 704 to 704 + n_labels (logits) [range of channel means: [-15, 20], range of channel variances: [15, 100]] / 
        # 3. 704 to 704 + n_labels (softmax probabilities) [range of channel values: [0,1]] 
    # KDEs of channels in the three groups need to be computed over a different range of values.
    # ==========================

    # ==============
    # placeholder for the points at which the PDFs are evaluated
    # ==============
    x_kde_g1_pl = tf.placeholder(tf.float32, shape = [61], name = 'x_kdes_g1') # shape [num_points_along_intensity_range]
    x_kde_g2_pl = tf.placeholder(tf.float32, shape = [101], name = 'x_kdes_g2') # shape [num_points_along_intensity_range]
    x_kde_g3_pl = tf.placeholder(tf.float32, shape = [101], name = 'x_kdes_g3') # shape [num_points_along_intensity_range]
    # ==============
    # placeholder for the smoothing factor in the KDE computation
    # ==============
    alpha_pl = tf.placeholder(tf.float32, shape = [], name = 'alpha') # shape [1]

    # ==============
    # Define ops for computing KDEs at each channel of each layer
    # ==============
    kdes_g1, kdes_g2, kdes_g3 = utils_kde.compute_1d_kdes(args.feature_subsampling_factor, # subsampling factor
                                                          args.features_randomized, # whether to sample randomly or uniformly
                                                          delta_x_pl, # zero padding information
                                                          delta_y_pl,
                                                          x_kde_g1_pl, # points where the KDEs have to be evaluated
                                                          x_kde_g2_pl,
                                                          x_kde_g3_pl,
                                                          alpha_pl) # smoothing parameter

    # ================================================================
    # add init ops
    # ================================================================
    init_ops = tf.global_variables_initializer()

    # ================================================================
    # Create session
    # ================================================================
    sess = tf.Session()

    # ================================================================
    # create saver
    # ================================================================
    saver_i2l = tf.train.Saver(var_list = i2l_vars)

    # ================================================================
    # freeze the graph before execution
    # ================================================================
    tf.get_default_graph().finalize()

    # ================================================================
    # Run the Op to initialize the variables.
    # ================================================================
    sess.run(init_ops)
    
    # ================================================================
    # Restore the segmentation network parameters
    # ================================================================
    logging.info('============================================================')   
    path_to_model = sys_config.project_root + 'log_dir/' + expname_i2l + 'models/'
    checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_dice.ckpt')
    logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
    saver_i2l.restore(sess, checkpoint_path)

    # ================================================================
    # Compute the KDEs for each SD subject
    # ================================================================
    kde_g1_params = [-3.0, 3.0, 0.1]
    kde_g2_params = [-5.0, 5.0, 0.1]
    kde_g3_params = [0.0, 1.0, 0.01]

    # [num_subjects, num_channels, num_x_points]
    sd_kdes_g1, sd_kdes_g2, sd_kdes_g3 = utils_kde.compute_sd_kdes(args.train_dataset,
                                                                   imtr,
                                                                   image_depth_tr,
                                                                   b_size_compute_sd_pdfs,
                                                                   sess,
                                                                   images_pl,
                                                                   alpha_pl,
                                                                   args.KDE_ALPHA,
                                                                   kdes_g1, kdes_g2, kdes_g3,
                                                                   x_kde_g1_pl, x_kde_g2_pl, x_kde_g3_pl,
                                                                   kde_g1_params, kde_g2_params, kde_g3_params,
                                                                   orig_data_res_x, # Arguments from here are to compute the padding for each subject
                                                                   orig_data_res_y, # so that, zero-padding, if any can be ignored while computing the PDFs.
                                                                   target_resolution,
                                                                   orig_data_siz_x,
                                                                   orig_data_siz_y,
                                                                   orig_data_siz_z,
                                                                   delta_x_pl,
                                                                   delta_y_pl,
                                                                   log_dir_pdfs)

    # ================================================================
    # close session
    # ================================================================
    sess.close()