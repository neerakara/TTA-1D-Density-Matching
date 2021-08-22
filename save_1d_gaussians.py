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
parser.add_argument('--train_dataset', default = "RUNMC") # RUNMC
parser.add_argument('--tr_run_number', type = int, default = 0) # 1
# Batch settings
parser.add_argument('--feature_subsampling_factor', type = int, default = 16) # 1 / 8 / 16
parser.add_argument('--features_randomized', type = int, default = 1) # 1 / 0
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
b_size_compute_sd_gaussians = dataset_params[10]

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
    images_pl = tf.placeholder(tf.float32, shape = [None] + list(image_size) + [1], name = 'images')
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
    # collect all vars in a list
    # ================================================================
    i2l_vars = []
    for v in tf.global_variables():
        var_name = v.name        
        i2l_vars.append(v)

    # ================================================================
    # placeholders for indicating amount of zero padding
    # ================================================================
    delta_x_pl = tf.placeholder(tf.int32, shape = [], name='zero_padding_x_pl')
    delta_y_pl = tf.placeholder(tf.int32, shape = [], name='zero_padding_y_pl')

    # ================================================================
    # Define ops for computing parameters of Gaussian distributions at each channel of each layer
    # ================================================================
    means1d, variances1d = utils_kde.compute_1d_gaussian_parameters(args.feature_subsampling_factor,
                                                                    args.features_randomized,
                                                                    delta_x_pl,
                                                                    delta_y_pl)
    
    # ================================================================
    # Add init ops
    # ================================================================
    init_ops = tf.global_variables_initializer()
            
    # ================================================================
    # Create session
    # ================================================================
    sess = tf.Session()

    # ================================================================
    # Create saver
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
    # Compute the SD Gaussian PDFs
    # ================================================================       
    sd_gaussians_fname = exp_config.make_sd_gaussian_names(log_dir_pdfs, args)

    gaussians_sd = utils_kde.compute_sd_gaussians(sd_gaussians_fname,
                                                  args.train_dataset,
                                                  imtr,            # Training images
                                                  image_depth_tr,  # Required to extract individual images for HCPT1
                                                  b_size_compute_sd_gaussians,
                                                  sess,
                                                  means1d,         # ops to compute Gaussian parameters.
                                                  variances1d,     
                                                  images_pl,       # image placeholder
                                                  orig_data_res_x, # Arguments from here are to compute the padding for each subject
                                                  orig_data_res_y, # so that, zero-padding, if any can be ignored while computing the PDFs.
                                                  target_resolution,
                                                  orig_data_siz_x,
                                                  orig_data_siz_y,
                                                  orig_data_siz_z,
                                                  delta_x_pl,
                                                  delta_y_pl)

    # ================================================================
    # close session
    # ================================================================
    sess.close()