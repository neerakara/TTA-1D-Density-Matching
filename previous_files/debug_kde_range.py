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
parser.add_argument('--test_dataset', default = "USZ") # PROMISE / USZ / CALTECH / STANFORD / HCPT2
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

    logging.info("I am here..") 

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
    orig_data_res_x = loaded_test_data[2]
    orig_data_res_y = loaded_test_data[3]
    orig_data_res_z = loaded_test_data[4]
    orig_data_siz_x = loaded_test_data[5]
    orig_data_siz_y = loaded_test_data[6]
    orig_data_siz_z = loaded_test_data[7]
    name_test_subjects = loaded_test_data[8]
    num_test_subjects = loaded_test_data[9]
    ids = loaded_test_data[10]
        
    # ================================================================
    # build the TF graph
    # ================================================================
    with tf.Graph().as_default():
        
        # ================================================================
        # create placeholders
        # ================================================================
        images_pl = tf.placeholder(tf.float32,
                                    shape = [None] + list(image_size) + [1],
                                    name = 'images')

        # ================================================================
        # insert a normalization module in front of the segmentation network
        # the normalization module is trained for each test image
        # ================================================================
        images_normalized, added_residual = model.normalize(images_pl,
                                                            exp_config,
                                                            training_pl = tf.constant(False, dtype=tf.bool))
        
        # ================================================================
        # build the graph that computes predictions from the inference model
        # ================================================================
        logits, softmax, preds = model.predict_i2l(images_normalized,
                                                   exp_config,
                                                   training_pl = tf.constant(False, dtype=tf.bool),
                                                   nlabels = nlabels)

        # ================================================================
        # Get features of one of the last layers and reduce their dimensionality with a random projection
        # ================================================================
        # last layer. From here, there is a 1x1 conv that gives the logits
        features_last_layer = tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + str(7) + '_' + str(2) + '_bn/FusedBatchNorm:0')
        patches_last_layer = utils_kde.extract_patches(features_last_layer,
                                                       channel = args.pca_channel,
                                                       psize = args.patch_size,
                                                       stride = args.pca_stride)

        # Accessing logits directly gives an error, but accesing it like this is fine! Weird TF!
        features_logits = tf.identity(logits)
        patches_logits = utils_kde.extract_patches(features_logits,
                                                       channel = args.pca_channel,
                                                       psize = args.patch_size,
                                                       stride = args.pca_stride)

        # ================================================================
        # divide the vars into segmentation network and normalization network
        # ================================================================
        i2l_vars = []
        normalization_vars = []
        bn_vars = []
        for v in tf.global_variables():
            var_name = v.name        
            i2l_vars.append(v)
            if 'image_normalizer' in var_name:
                normalization_vars.append(v)
            if 'beta' in var_name or 'gamma' in var_name:
                bn_vars.append(v)

        if args.tta_vars == 'BN':
            tta_vars = bn_vars
        elif args.tta_vars == 'NORM':
            tta_vars = normalization_vars
                                
        # ================================================================
        # add init ops
        # ================================================================
        init_ops = tf.global_variables_initializer()
                
        # ================================================================
        # create session
        # ================================================================
        sess = tf.Session()

        # ================================================================
        # create saver
        # ================================================================
        saver_i2l = tf.train.Saver(var_list = i2l_vars)
        saver_tta = tf.train.Saver(var_list = tta_vars)        
                    
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
        path_to_model = log_dir_sd + 'models/'
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_dice.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_i2l.restore(sess, checkpoint_path)

        # ================================================================
        # Create dir for PCA
        # ================================================================
        # prefix = 'pca/p' + str(args.patch_size) + 's' + str(args.pca_stride) + '_dim' + str(args.PCA_LATENT_DIM) + '_' + args.pca_layer + '_channel' + str(args.pca_channel) + '/'
        # if not tf.gfile.Exists(log_dir_sd + prefix):
        #     tf.gfile.MakeDirs(log_dir_sd + prefix)

        # ================================================================
        # Extract features from the last layer (before the logits) of all SD images
        # ================================================================
        sd_features = np.zeros([orig_data_siz_z_train.shape[0], image_size[0], image_size[1]]) 
        sd_patches = np.zeros([1,args.patch_size*args.patch_size])
        feat_stats_sd = []

        for train_sub_num in range(orig_data_siz_z_train.shape[0]):
            train_image = imtr[np.sum(orig_data_siz_z_train[:train_sub_num]) : np.sum(orig_data_siz_z_train[:train_sub_num+1]),:,:]
            feed_dict={images_pl: np.expand_dims(train_image, axis=-1)}
            if args.pca_layer == 'layer_7_2':
                feats_last_layer = sess.run(features_last_layer, feed_dict=feed_dict)
                ptchs_last_layer = sess.run(patches_last_layer, feed_dict=feed_dict)
            elif args.pca_layer == 'logits':
                feats_last_layer = sess.run(features_logits, feed_dict=feed_dict)
                ptchs_last_layer = sess.run(patches_logits, feed_dict=feed_dict)

            sd_features[train_sub_num, :, :] = feats_last_layer[feats_last_layer.shape[0]//2, :, :, args.pca_channel]
            sd_patches = np.concatenate((sd_patches, ptchs_last_layer), axis=0)
            logging.info("Number of patches in SD subject " + str(train_sub_num+1) + ": " + str(ptchs_last_layer.shape[0]))

            # For this subject, for each channel of the features,
            # print stats of the intensity values
            feat_stats_this_subject = []
            logging.info("============================== subject: " + str(train_sub_num))
            for c in range(feats_last_layer.shape[-1]):
                stats_this_channel = []
                stats_this_channel.append(np.round(np.min(feats_last_layer[:, :, :, c]), 2))
                stats_this_channel.append(np.round(np.max(feats_last_layer[:, :, :, c]), 2))
                stats_this_channel.append(np.round(np.percentile(feats_last_layer[:, :, :, c], 0.1), 2))
                stats_this_channel.append(np.round(np.percentile(feats_last_layer[:, :, :, c], 99.9), 2))
                feat_stats_this_subject.append(np.array(stats_this_channel))
            
            feat_stats_sd.append(np.array(feat_stats_this_subject))

        feat_stats_sd = np.array(feat_stats_sd)

        for c in range(feat_stats_sd.shape[1]):
            logging.info('channel ' + str(c))
            logging.info('Mean, std deviation across SD subjects of 0.1%tile value: ' + str(np.round(np.mean(feat_stats_sd[:,c,2]), 1)) + ', ' + str(np.round(np.std(feat_stats_sd[:,c,2]), 1)))
            logging.info('Mean, std deviation across SD subjects of 99.9%tile value: ' + str(np.round(np.mean(feat_stats_sd[:,c,3]), 1)) + ', ' + str(np.round(np.std(feat_stats_sd[:,c,3]), 1)))

        # sd_patches = np.array(sd_patches[1:,:])
        # logging.info("Number of all SD patches:" + str(sd_patches.shape[0]))
        # random_indices = np.random.randint(0, sd_patches.shape[0], 100)

        # redraw = True
        # # visualize feature maps 
        # if (redraw == True) or (tf.gfile.Exists(log_dir_sd + prefix + 'SD_features.png') == False):
        #     utils_vis.save_features(sd_features, savepath = log_dir_sd + prefix + 'SD_features.png')
        
        # # visualize some patches (randomly selected from all SD subjects)
        # if (redraw == True) or (tf.gfile.Exists(log_dir_sd + prefix + 'patches_NCI.png') == False):
        #     utils_vis.save_patches(sd_patches,
        #                            savepath = log_dir_sd + prefix + 'patches_NCI.png',
        #                            ids = random_indices,
        #                            nc = 10,
        #                            nr = 10,
        #                            psize = args.patch_size)

# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()