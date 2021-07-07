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
parser.add_argument('--alpha', type = float, default = 10.0) # 10.0 / 100.0 / 1000.0
# Which vars to adapt?
parser.add_argument('--tta_vars', default = "NORM") # BN / NORM
# How many moments to match and how?
parser.add_argument('--match_moments', default = "All_KL") # Gaussian_KL / All_KL / All_CF_L2
parser.add_argument('--before_or_after_bn', default = "AFTER") # AFTER / BEFORE
# PCA settings
parser.add_argument('--patch_size', type = int, default = 16) # 32 / 64 / 128
parser.add_argument('--pca_stride', type = int, default = 8) # 64 / 128
parser.add_argument('--pca_layer', default = 'layer_7_2') # layer_7_2 / logits / softmax
parser.add_argument('--PCA_LATENT_DIM', type = int, default = 10) # 10 / 50
parser.add_argument('--pca_kde_alpha', type = float, default = 10.0) # 0.1 / 1.0 / 10.0
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
        pca_channel_pl = tf.placeholder(tf.int32, shape = [], name = 'channel_num')
        # last layer. From here, there is a 1x1 conv that gives the logits
        features_last_layer = tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + str(7) + '_' + str(2) + '_bn/FusedBatchNorm:0')
        patches_last_layer = utils_kde.extract_patches(features_last_layer, pca_channel_pl, args.patch_size, args.pca_stride)

        # Accessing logits directly gives an error, but accesing it like this is fine! Weird TF!
        features_logits = tf.identity(logits)
        patches_logits = utils_kde.extract_patches(features_logits, pca_channel_pl, args.patch_size, args.pca_stride)

        # Get softmax and use these to select active patches
        features_softmax = tf.identity(softmax)
        patches_softmax = utils_kde.extract_patches(features_softmax,
                                                    channel = 2, # This needs to be 1 or 2 for prostate data
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
        prefix = 'pca/p' + str(args.patch_size) 
        prefix = prefix + 's' + str(args.pca_stride)
        prefix = prefix + '_dim' + str(args.PCA_LATENT_DIM)
        prefix = prefix + '_' + args.pca_layer + '/'
        if not tf.gfile.Exists(log_dir_sd + prefix):
            tf.gfile.MakeDirs(log_dir_sd + prefix)

        # ================================================================
        # Do Channel-wise PCA
        # ================================================================
        if args.pca_layer == 'layer_7_2':
            num_channels = 16
        else: # 'logits' / 'softmax'
            num_channels = nlabels

        for channel in range(2, num_channels):

            logging.info("==================================")
            logging.info("Channel " + str(channel))

            # init arrays to store features and patches of different subjects
            sd_features = np.zeros([orig_data_siz_z_train.shape[0], image_size[0], image_size[1]]) 
            sd_patches = np.zeros([1,args.patch_size*args.patch_size])
            sd_patches_active = np.zeros([1,args.patch_size*args.patch_size])

            # go through all sd subjects
            # extract features and patches
            for train_sub_num in range(orig_data_siz_z_train.shape[0]):
                train_image = imtr[np.sum(orig_data_siz_z_train[:train_sub_num]) : np.sum(orig_data_siz_z_train[:train_sub_num+1]),:,:]
                feed_dict={images_pl: np.expand_dims(train_image, axis=-1), pca_channel_pl: channel}
                if args.pca_layer == 'layer_7_2':
                    feats_last_layer = sess.run(features_last_layer, feed_dict=feed_dict)
                    ptchs_last_layer = sess.run(patches_last_layer, feed_dict=feed_dict)
                elif args.pca_layer == 'logits':
                    feats_last_layer = sess.run(features_logits, feed_dict=feed_dict)
                    ptchs_last_layer = sess.run(patches_logits, feed_dict=feed_dict)
                elif args.pca_layer == 'softmax':
                    feats_last_layer = sess.run(features_softmax, feed_dict=feed_dict)
                    ptchs_last_layer = sess.run(patches_softmax, feed_dict=feed_dict)
                
                # get corresponding patches of the softmax of class 2
                ptchs_last_layer_softmax = sess.run(patches_softmax, feed_dict=feed_dict)

                # collect features from the central slice for vis
                sd_features[train_sub_num, :, :] = feats_last_layer[feats_last_layer.shape[0]//2, :, :, channel]

                # number of patches from this subject
                logging.info("Number of patches in SD subject " + str(train_sub_num+1) + ": " + str(ptchs_last_layer.shape[0]))
                sd_patches = np.concatenate((sd_patches, ptchs_last_layer), axis=0)

                # extract 'active' patches -> ones for which the softmax of class two has a high value in the central pixel
                actives_ptchs_last_layer = ptchs_last_layer[np.where(ptchs_last_layer_softmax[:, (args.patch_size * (args.patch_size + 1))//2] > 0.8)[0], :]
                logging.info("Number of active patches in SD subject " + str(train_sub_num+1) + ": " + str(actives_ptchs_last_layer.shape[0]))
                sd_patches_active = np.concatenate((sd_patches_active, actives_ptchs_last_layer), axis=0)

            # remove dummy patch added in the front, before the loop over all sd subjects
            sd_patches = np.array(sd_patches[1:,:])
            sd_patches_active = np.array(sd_patches_active[1:,:])
            logging.info("Number of all SD patches:" + str(sd_patches.shape[0]))
            logging.info("Number of all active SD patches:" + str(sd_patches_active.shape[0]))

            redraw = False
            # visualize feature maps 
            if (redraw == True) or (tf.gfile.Exists(log_dir_sd + prefix + 'SD_features_c' + str(channel) + '.png') == False):
                utils_vis.save_features(sd_features, savepath = log_dir_sd + prefix + 'SD_features_c' + str(channel) + '.png')
            
            # visualize some patches (randomly selected from all SD subjects)
            if (redraw == True) or (tf.gfile.Exists(log_dir_sd + prefix + 'SD_patches_c' + str(channel) + '.png') == False):
                random_indices = np.random.randint(0, sd_patches.shape[0], 100)
                utils_vis.save_patches(sd_patches,
                                    savepath = log_dir_sd + prefix + 'SD_patches_c' + str(channel) + '.png',
                                    ids = random_indices,
                                    nc = 5,
                                    nr = 5,
                                    psize = args.patch_size)

                random_indices_active = np.random.randint(0, sd_patches_active.shape[0], 100)
                utils_vis.save_patches(sd_patches_active,
                                    savepath = log_dir_sd + prefix + 'SD_active_patches_c' + str(channel) + '.png',
                                    ids = random_indices_active,
                                    nc = 5,
                                    nr = 5,
                                    psize = args.patch_size)

            # PCA
            pca_path = log_dir_sd + prefix + 'pca_sd_c' + str(channel) + '.pkl'
            if not os.path.exists(pca_path):
                num_pcs = args.PCA_LATENT_DIM
                pca = PCA(n_components = num_pcs, whiten=True)
                logging.info("Fitting PCA parameters to centered SD patches from all subjects")
                pca.fit(sd_patches_active)
                # The fit method subtracts the mean of each feature and saves the subtracted mean
                # The mean is then used while transforming new datapoints from x to z or z to x.
                # write to file
                pk.dump(pca, open(pca_path, "wb"))
            else:
                logging.info("PCA already done. Loading saved model..")
                pca = pk.load(open(pca_path, 'rb'))
                num_pcs = pca.n_components_

            # how much variance is explained by the first k components
            logging.info("sum of ratios of variance explained by the first " + str(num_pcs) + " PCs:")
            logging.info(np.round(np.sum(pca.explained_variance_ratio_), 3))

            # visualize the principal components
            if (redraw == True) or (tf.gfile.Exists(log_dir_sd + prefix + 'pcs_c' + str(channel) + '.png') == False):
                logging.info("Visualizing principal components..")
                utils_vis.visualize_principal_components(pca.components_, log_dir_sd + prefix + 'pcs_c' + str(channel) + '.png', args.patch_size, nc = 2, nr = 2)

            # plot scatter plots of pairs of latent dimensions for patches of each SD subject      
            logging.info("Visualizing pairwise scatter plots of latent dimensions of individual SD subjects..")
            kdes_all_sd_subjects = []
            for train_sub_num in range(orig_data_siz_z_train.shape[0]):
                logging.info("SD subject " + str(train_sub_num+1))
                train_image = imtr[np.sum(orig_data_siz_z_train[:train_sub_num]) : np.sum(orig_data_siz_z_train[:train_sub_num+1]),:,:]
                feed_dict={images_pl: np.expand_dims(train_image, axis=-1), pca_channel_pl: channel}
                sd_patches_this_sub = sess.run(patches_last_layer, feed_dict = feed_dict)
                sd_patches_this_sub_softmax = sess.run(patches_softmax, feed_dict=feed_dict)
                sd_active_patches_this_sub = sd_patches_this_sub[np.where(sd_patches_this_sub_softmax[:, (args.patch_size * (args.patch_size + 1))//2] > 0.8)[0], :]
                sd_active_patches_this_sub_latent = pca.transform(sd_active_patches_this_sub)
                logging.info("Min latent value: " + str(np.round(np.min(sd_active_patches_this_sub_latent), 2)))
                logging.info("Max latent value: " + str(np.round(np.max(sd_active_patches_this_sub_latent), 2)))

                # compute dimension wise KDE for this subject
                kdes_this_subject, z_vals = utils_kde.compute_pca_latent_kdes(sd_active_patches_this_sub_latent, args.pca_kde_alpha)
                kdes_all_sd_subjects.append(kdes_this_subject)

            kdes_all_sd_subjects = np.array(kdes_all_sd_subjects)
            kde_path = log_dir_sd + prefix + 'kde_alpha' + str(args.pca_kde_alpha) + '_c' + str(channel) + '.npy'
            np.save(kde_path, kdes_all_sd_subjects)

            utils_vis.plot_kdes_for_sd_latents(kdes_all_sd_subjects,
                                               z_vals,
                                               savepath = log_dir_sd + prefix + 'kde_c' + str(channel) + '.png')

        # ================================================================
        # Close the session
        # ================================================================
        sess.close()
        
# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()
