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
parser.add_argument('--alpha', type = float, default = 10.0) # 10.0 / 100.0 / 1000.0
# Which vars to adapt?
parser.add_argument('--tta_vars', default = "NORM") # BN / NORM
# How many moments to match and how?
parser.add_argument('--match_moments', default = "All_KL") # Gaussian_KL / All_KL / All_CF_L2
parser.add_argument('--before_or_after_bn', default = "AFTER") # AFTER / BEFORE
# PCA settings
parser.add_argument('--patch_size', type = int, default = 8) # 32 / 64 / 128
parser.add_argument('--pca_stride', type = int, default = 8) # 64 / 128
parser.add_argument('--pca_layer', default = 'layer_7_2') # layer_7_2 / logits / softmax
parser.add_argument('--PCA_LATENT_DIM', type = int, default = 10) # 10 / 50
parser.add_argument('--pca_kde_alpha', type = float, default = 10.0) # 0.1 / 1.0 / 10.0
parser.add_argument('--PCA_TR_NUM', type = int, default = 15) # 10 / 15
# The PCA will be trained on the first 'PCA_TR_NUM' images from the CNN training dataset
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
    # load validation images
    # ===================================
    imvl, gtvl, orig_data_siz_z_val, num_val_subjects = utils.load_validation_data(args.train_dataset,
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

        # Combine the softmax scores into a map of foreground probabilities and use this to select active patches
        features_fg_probs = tf.expand_dims(tf.math.reduce_max(softmax[:, :, :, 1:], axis=-1), axis=-1)
        patches_fg_probs = utils_kde.extract_patches(features_fg_probs,
                                                     channel = 0, 
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
        prefix = prefix + '_' + args.pca_layer + '_active_all_fg_numtr' + str(args.PCA_TR_NUM) + '/'
        if not tf.gfile.Exists(log_dir_sd + prefix):
            tf.gfile.MakeDirs(log_dir_sd + prefix)

        # ================================================================
        # Do Channel-wise PCA
        # ================================================================
        if args.pca_layer == 'layer_7_2':
            num_channels = 16
        else: # 'logits' / 'softmax'
            num_channels = nlabels

        # 
        kl_trtt_sdtd = []
        kl_trvl_sdtd = []
        kl_trts_sdtd = []
        kl_trtt_tdsd = []
        kl_trvl_tdsd = []
        kl_trts_tdsd = []

        # go through all channels of this feature layer
        for channel in range(num_channels):

            logging.info("==================================")
            logging.info("Channel " + str(channel))

            # init arrays to store features and patches of different subjects
            sd_features = np.zeros([args.PCA_TR_NUM, image_size[0], image_size[1]]) 
            sd_patches = np.zeros([1,args.patch_size*args.patch_size])
            sd_patches_active = np.zeros([1,args.patch_size*args.patch_size])

            # go through all sd subjects
            # extract features and patches
            for train_sub_num in range(args.PCA_TR_NUM):
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
                
                # get corresponding patches of the foreground probability values
                ptchs_fg_probs = sess.run(patches_fg_probs, feed_dict=feed_dict)

                # collect features from the central slice for vis
                sd_features[train_sub_num, :, :] = feats_last_layer[feats_last_layer.shape[0]//2, :, :, channel]

                # number of patches from this subject
                # logging.info("Number of patches in SD subject " + str(train_sub_num+1) + ": " + str(ptchs_last_layer.shape[0]))
                sd_patches = np.concatenate((sd_patches, ptchs_last_layer), axis=0)

                # extract 'active' patches -> ones for which the softmax of class two has a high value in the central pixel
                active_ptchs_last_layer = ptchs_last_layer[np.where(ptchs_fg_probs[:, (args.patch_size * (args.patch_size + 1))//2] > 0.8)[0], :]
                # logging.info("Number of active patches in SD subject " + str(train_sub_num+1) + ": " + str(active_ptchs_last_layer.shape[0]))
                sd_patches_active = np.concatenate((sd_patches_active, active_ptchs_last_layer), axis=0)

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
            logging.info("sum of ratios of variance explained by the first " + str(num_pcs) + " PCs:" + str(np.round(np.sum(pca.explained_variance_ratio_), 2)))

            # visualize the principal components
            if (redraw == True) or (tf.gfile.Exists(log_dir_sd + prefix + 'pcs_c' + str(channel) + '.png') == False):
                # logging.info("Visualizing principal components..")
                utils_vis.visualize_principal_components(pca.components_, log_dir_sd + prefix + 'pcs_c' + str(channel) + '.png', args.patch_size, nc = 2, nr = 2)

            # Compute KDEs of each latent dimension for patches of each SD training subject      
            # logging.info("Computing KDEs in each latent dimension for individual SD training subjects..")
            num_tr_slices = np.sum(orig_data_siz_z_train[:args.PCA_TR_NUM])
            kdes_all_sd_tr_subs, z_vals, feats_tr, act_pats_tr = utils_kde.compute_latent_kdes_subjectwise(images = imtr[:num_tr_slices, :, :],
                                                                                                 image_size = image_size,
                                                                                                 image_depths = orig_data_siz_z_train[:args.PCA_TR_NUM],
                                                                                                 image_placeholder = images_pl,
                                                                                                 channel_placeholder = pca_channel_pl,
                                                                                                 channel_num = channel,
                                                                                                 features = features_last_layer,
                                                                                                 patches = patches_last_layer,
                                                                                                 fg_probs = patches_fg_probs,
                                                                                                 psize = args.patch_size,
                                                                                                 threshold = 0.8,
                                                                                                 learned_pca = pca,
                                                                                                 alpha_kde = args.pca_kde_alpha,
                                                                                                 sess = sess,
                                                                                                 savepath = log_dir_sd + prefix + 'kde_alpha' + str(args.pca_kde_alpha) + '_c' + str(channel) + '.npy')
            
            # Compute KDEs for the images that were part of the CNN training, but not part of the PCA training
            if args.PCA_TR_NUM < orig_data_siz_z_train.shape[0]:
                kdes_all_sd_tt_subs, z_vals, feats_tt, act_pats_tt = utils_kde.compute_latent_kdes_subjectwise(images = imtr[num_tr_slices:, :, :],
                                                                                                    image_size = image_size,
                                                                                                    image_depths = orig_data_siz_z_train[args.PCA_TR_NUM:],
                                                                                                    image_placeholder = images_pl,
                                                                                                    channel_placeholder = pca_channel_pl,
                                                                                                    channel_num = channel,
                                                                                                    features = features_last_layer,
                                                                                                    patches = patches_last_layer,
                                                                                                    fg_probs = patches_fg_probs,
                                                                                                    psize = args.patch_size,
                                                                                                    threshold = 0.8,
                                                                                                    learned_pca = pca,
                                                                                                    alpha_kde = args.pca_kde_alpha,
                                                                                                    sess = sess)
            else:
                kdes_all_sd_tt_subs = kdes_all_sd_tr_subs
                feats_tt = feats_tr
                act_pats_tt = act_pats_tr


            # KDEs of images that were neither part of the CNN training, nor the PCA training, but come from the same distribution at the SD images
            kdes_all_sd_vl_subs, z_vals, feats_vl, act_pats_vl = utils_kde.compute_latent_kdes_subjectwise(images = imvl,
                                                                                                 image_size = image_size,
                                                                                                 image_depths = orig_data_siz_z_val,
                                                                                                 image_placeholder = images_pl,
                                                                                                 channel_placeholder = pca_channel_pl,
                                                                                                 channel_num = channel,
                                                                                                 features = features_last_layer,
                                                                                                 patches = patches_last_layer,
                                                                                                 fg_probs = patches_fg_probs,
                                                                                                 psize = args.patch_size,
                                                                                                 threshold = 0.8,
                                                                                                 learned_pca = pca,
                                                                                                 alpha_kde = args.pca_kde_alpha,
                                                                                                 sess = sess)

            # KDEs of images from TD
            kdes_all_td_ts_subs, z_vals, feats_ts, act_pats_ts = utils_kde.compute_latent_kdes_subjectwise(images = imts,
                                                                                                 image_size = image_size,
                                                                                                 image_depths = orig_data_siz_z,
                                                                                                 image_placeholder = images_pl,
                                                                                                 channel_placeholder = pca_channel_pl,
                                                                                                 channel_num = channel,
                                                                                                 features = features_last_layer,
                                                                                                 patches = patches_last_layer,
                                                                                                 fg_probs = patches_fg_probs,
                                                                                                 psize = args.patch_size,
                                                                                                 threshold = 0.8,
                                                                                                 learned_pca = pca,
                                                                                                 alpha_kde = args.pca_kde_alpha,
                                                                                                 sess = sess)

            # logging.info('features in training (CNN) and training (PCA): ' + str(feats_tr.shape))
            # logging.info('features in training (CNN), but testing (PCA): ' + str(feats_tt.shape))
            # logging.info('features in validation (5 subs): ' + str(feats_vl.shape))
            # logging.info('features in testing (20 subs): ' + str(feats_ts.shape))

            # utils_vis.save_features(feats_tr, savepath = log_dir_sd + prefix + 'features_tr_c' + str(channel) + '.png')
            # utils_vis.save_features(feats_tt, savepath = log_dir_sd + prefix + 'features_tt_c' + str(channel) + '.png')
            # utils_vis.save_features(feats_vl, savepath = log_dir_sd + prefix + 'features_vl_c' + str(channel) + '.png')
            # utils_vis.save_features(feats_ts, savepath = log_dir_sd + prefix + 'features_ts_c' + str(channel) + '.png')

            # logging.info('number of active patches in training (CNN) and training (PCA): ' + str(act_pats_tr.shape))
            # logging.info('number of active patches in training (CNN), but testing (PCA): ' + str(act_pats_tt.shape))
            # logging.info('number of active patches in validation (5 subs): ' + str(act_pats_vl.shape))
            # logging.info('number of active patches in testing (20 subs): ' + str(act_pats_ts.shape))

            # utils_vis.save_patches(act_pats_tr,
            #                        savepath = log_dir_sd + prefix + 'act_pats_tr_c' + str(channel) + '.png',
            #                        ids = np.random.randint(0, act_pats_tr.shape[0], 25),
            #                        nc = 5,
            #                        nr = 5,
            #                        psize = args.patch_size)
            # utils_vis.save_patches(act_pats_tt,
            #                        savepath = log_dir_sd + prefix + 'act_pats_tt_c' + str(channel) + '.png',
            #                        ids = np.random.randint(0, act_pats_tt.shape[0], 25),
            #                        nc = 5,
            #                        nr = 5,
            #                        psize = args.patch_size)
            # utils_vis.save_patches(act_pats_vl,
            #                        savepath = log_dir_sd + prefix + 'act_pats_vl_c' + str(channel) + '.png',
            #                        ids = np.random.randint(0, act_pats_vl.shape[0], 25),
            #                        nc = 5,
            #                        nr = 5,
            #                        psize = args.patch_size)
            # utils_vis.save_patches(act_pats_ts,
            #                        savepath = log_dir_sd + prefix + 'act_pats_ts_c' + str(channel) + '.png',
            #                        ids = np.random.randint(0, act_pats_ts.shape[0], 25),
            #                        nc = 5,
            #                        nr = 5,
            #                        psize = args.patch_size)

            # compute average KL between across all pairs of KDEs
            avg_kl_trtt_sdtd, avg_kl_trtt_tdsd = utils_kde.compute_kl_between_kdes_numpy(kdes_all_sd_tr_subs, kdes_all_sd_tt_subs)
            avg_kl_trvl_sdtd, avg_kl_trvl_tdsd = utils_kde.compute_kl_between_kdes_numpy(kdes_all_sd_tr_subs, kdes_all_sd_vl_subs)
            avg_kl_trts_sdtd, avg_kl_trts_tdsd = utils_kde.compute_kl_between_kdes_numpy(kdes_all_sd_tr_subs, kdes_all_td_ts_subs)

            kl_trtt_sdtd.append(avg_kl_trtt_sdtd)
            kl_trvl_sdtd.append(avg_kl_trvl_sdtd)
            kl_trts_sdtd.append(avg_kl_trts_sdtd)
            kl_trtt_tdsd.append(avg_kl_trtt_tdsd)
            kl_trvl_tdsd.append(avg_kl_trvl_tdsd)
            kl_trts_tdsd.append(avg_kl_trts_tdsd)

            # plot KDEs of SD train, SD val and TD test subjects
            utils_vis.plot_kdes_for_latents(kdes_all_sd_tr_subs,
                                            kdes_all_sd_tt_subs,
                                            kdes_all_sd_vl_subs,
                                            kdes_all_td_ts_subs,
                                            z_vals,
                                            log_dir_sd + prefix + 'kde_c' + str(channel) + '.png')

        logging.info("MEAN KL (over channels, subject pairs, latent dims) TR vs TT, SDvTD: " + str(np.mean(np.array(kl_trtt_sdtd))))
        logging.info("MEAN KL (over channels, subject pairs, latent dims) TR vs VL, SDvTD: " + str(np.mean(np.array(kl_trvl_sdtd))))
        logging.info("MEAN KL (over channels, subject pairs, latent dims) TR vs TS, SDvTD: " + str(np.mean(np.array(kl_trts_sdtd))))
        logging.info("MEAN KL (over channels, subject pairs, latent dims) TR vs TT, TDvSD: " + str(np.mean(np.array(kl_trtt_tdsd))))
        logging.info("MEAN KL (over channels, subject pairs, latent dims) TR vs VL, TDvSD: " + str(np.mean(np.array(kl_trvl_tdsd))))
        logging.info("MEAN KL (over channels, subject pairs, latent dims) TR vs TS, TDvSD: " + str(np.mean(np.array(kl_trts_tdsd))))

        # ================================================================
        # Close the session
        # ================================================================
        sess.close()
        
# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()
