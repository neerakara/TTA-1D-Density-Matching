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
# MRF settings
parser.add_argument('--BINARY', default = 1) # 1 / 0
parser.add_argument('--POTENTIAL_TYPE', type = int, default = 2) # 1 / 2
parser.add_argument('--BINARY_LAMBDA', type = float, default = 0.1) # 1.0
parser.add_argument('--BINARY_ALPHA', type = float, default = 1.0) # 1.0 / 10.0 (smoothness paramter for the KDE of the binary potentials)
# PCA settings
parser.add_argument('--patch_size', type = int, default = 32) # 32 / 64 / 128
parser.add_argument('--pca_stride', type = int, default = 16) # 64 / 128
parser.add_argument('--pca_channel', type = int, default = 0) # 0 / 1 .. 15
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
# Make the name for this TTA run
# ================================================================
exp_str = exp_config.make_tta_exp_name(args)

# ================================================================
# Setup directories for this run
# ================================================================
expname_i2l = 'tr' + args.train_dataset + '_r' + str(args.tr_run_number) + '/' + 'i2i2l/'
log_dir_sd = sys_config.project_root + 'log_dir/' + expname_i2l
log_dir_tta = log_dir_sd + exp_str

# ==================================================================
# Identifier for SFDA
# ==================================================================
if args.TTA_or_SFDA == 'SFDA':
    if args.test_dataset == 'USZ':
        td_string = 'SFDA_' + args.test_dataset + '/'
    elif args.test_dataset == 'PROMISE':    
        td_string = 'SFDA_' + args.test_dataset + '_' + args.PROMISE_SUB_DATASET + '/'

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
        # last layer. From here, there is a 1x1 conv that gives the logits
        conv_string = str(7) + '_' + str(2)
        features_last_layer = tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '_bn/FusedBatchNorm:0')
        patches_last_layer = utils_kde.extract_patches(features_last_layer,
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

        prefix = 'pca/p' + str(args.patch_size) + 's' + str(args.pca_stride) + '/'
        if not tf.gfile.Exists(log_dir_sd + prefix):
            tf.gfile.MakeDirs(log_dir_sd + prefix)

        # ================================================================
        # Extract features from the last layer (before the logits) of all SD images
        # ================================================================
        sd_patches = np.zeros([1,args.patch_size*args.patch_size])
        for train_sub_num in range(orig_data_siz_z_train.shape[0]):
            train_image = imtr[np.sum(orig_data_siz_z_train[:train_sub_num]) : np.sum(orig_data_siz_z_train[:train_sub_num+1]),:,:]
            feed_dict={images_pl: np.expand_dims(train_image, axis=-1)}
            feats_last_layer = sess.run(features_last_layer, feed_dict=feed_dict)
            ptchs_last_layer = sess.run(patches_last_layer, feed_dict=feed_dict)
            sd_patches = np.concatenate((sd_patches, ptchs_last_layer), axis=0)
            logging.info("Number of patches in SD subject " + str(train_sub_num+1) + ": " + str(ptchs_last_layer.shape[0]))

        sd_patches = np.array(sd_patches[1:,:])
        logging.info("Number of all SD patches:" + str(sd_patches.shape[0]))
        random_indices = np.random.randint(0, sd_patches.shape[0], 100)

        redraw = False
        # visualize feature maps 
        if (redraw == True) or (tf.gfile.Exists(log_dir_sd + prefix) == False):
            utils_vis.save_features(feats_last_layer, savepath = log_dir_sd + prefix + 'features_NCI_onesubect.png')
        
        # visualize some patches (randomly selected from all SD subjects)
        if (redraw == True) or (tf.gfile.Exists(log_dir_sd + prefix + 'patches_NCI.png') == False):
            utils_vis.save_patches(sd_patches,
                                savepath = log_dir_sd + prefix + 'patches_NCI.png',
                                ids = random_indices,
                                nc = 10,
                                nr = 10,
                                psize = args.patch_size)

        # PCA
        pca_path = log_dir_sd + prefix + 'pca_all_sd_subjects_channel' + str(args.pca_channel) + '.pkl'
        if not os.path.exists(pca_path):
            num_pcs = 50
            pca = PCA(n_components = num_pcs, whiten=True)
            logging.info("Fitting PCA parameters to centered SD patches from all subjects")
            pca.fit(sd_patches)
            # The fit method subtracts the mean of each feature and saves the subtracted mean
            # The mean is then used while transforming new datapoints from x to z or z to x.
            # write to file
            pk.dump(pca, open(pca_path, "wb"))
        else:
            logging.info("PCA already done. Loading saved model..")
            pca = pk.load(open(pca_path, 'rb'))
            num_pcs = pca.n_components_

        # how much variance is explained by the first k components
        logging.info("sum of ratios of variance explained by the first 10 PCs:")
        logging.info(np.round(np.sum(pca.explained_variance_ratio_[:10]), 3))
        logging.info("sum of ratios of variance explained by the first " + str(num_pcs) + " PCs:")
        logging.info(np.round(np.sum(pca.explained_variance_ratio_), 3))

        # how much variance is explained by the first k components
        logging.info("Variance explained by the first 10 PCs (eigenvalues of covariance matrix):")
        logging.info(np.round((pca.explained_variance_[:10]), 3))

        # visualize the principal components
        if (redraw == True) or (tf.gfile.Exists(log_dir_sd + prefix + 'principal_components.png') == False):
            logging.info("Visualizing principal components..")
            utils_vis.visualize_principal_components(pca.components_, log_dir_sd + prefix + 'principal_components.png', args.patch_size, nc = 5, nr = 5)

        # get latent representation of all SD patches (of all subjects)
        logging.info("Finding latent representation of SD patches:")
        sd_patches_latent = pca.transform(sd_patches)
        logging.info("size of latent representation of SD patches: " + str(sd_patches_latent.shape))

        # plot scatter plot of pairs of latent dimensions of all SD patches
        if (redraw == True) or (tf.gfile.Exists(log_dir_sd + prefix + 'pca_coefs_whitened_sd_pairwise.png') == False):
            logging.info("Visualizing pairwise scatter plots of latent dimensions of all SD subjects..")
            utils_vis.plot_scatter_pca_coefs_pairwise(sd_patches_latent,
                                                    savepath = log_dir_sd + prefix + 'pca_coefs_whitened_sd_pairwise.png')

        # visualize reconstructions of all sd patches
        sd_patches_recon = pca.inverse_transform(sd_patches_latent)
        if (redraw == True) or (tf.gfile.Exists(log_dir_sd + prefix + 'patches_NCI_recon.png') == False):
            utils_vis.save_patches(sd_patches_recon,
                                   savepath = log_dir_sd + prefix + 'patches_NCI_recon.png',
                                   ids = random_indices,
                                   nc = 10,
                                   nr = 10,
                                   psize = args.patch_size)

        # plot scatter plots of pairs of latent dimensions for patches of each SD subject      
        logging.info("Visualizing pairwise scatter plots of latent dimensions of individual SD subjects..")
        kdes_all_sd_subjects = []
        for train_sub_num in range(orig_data_siz_z_train.shape[0]):
            logging.info("SD subject " + str(train_sub_num+1))
            train_image = imtr[np.sum(orig_data_siz_z_train[:train_sub_num]) : np.sum(orig_data_siz_z_train[:train_sub_num+1]),:,:]
            feed_dict={images_pl: np.expand_dims(train_image, axis=-1)}
            sd_patches_this_sub = sess.run(patches_last_layer, feed_dict = feed_dict)
            sd_patches_this_sub_latent = pca.transform(sd_patches_this_sub)

            # compute dimension wise KDE for this subject
            kdes_this_subject, z_vals = utils_kde.compute_pca_latent_kdes(sd_patches_this_sub_latent, args.pca_kde_alpha)
            kdes_all_sd_subjects.append(kdes_this_subject)

            if (redraw == True) or (tf.gfile.Exists(log_dir_sd + prefix + 'pca_coefs_sd_pairwise_sub' + str(train_sub_num+1) + '.png') == False):
                utils_vis.plot_scatter_pca_coefs_pairwise(sd_patches_this_sub_latent,
                                                          savepath = log_dir_sd + prefix + 'pca_coefs_sd_pairwise_sub' + str(train_sub_num+1) + '.png')

            if (redraw == True) or (tf.gfile.Exists(log_dir_sd + prefix + 'kde_pca_coefs_sd_sub' + str(train_sub_num+1) + '.png') == False):
                utils_vis.plot_histograms_pca_coefs(kdes_this_subject,
                                                    z_vals,
                                                    savepath = log_dir_sd + prefix + 'kde_pca_coefs_sd_sub' + str(train_sub_num+1) + '.png')

        kdes_all_sd_subjects = np.array(kdes_all_sd_subjects)
        kde_path = log_dir_sd + prefix + 'kde_alpha' + str(args.pca_kde_alpha) + '_all_sd_subjects_channel' + str(args.pca_channel) + '.npy'
        np.save(kde_path, kdes_all_sd_subjects)

        # ==================================================================
        # Check if the KDEs of TD subjects differ from those of the SD subjects
        # ==================================================================
        for sub_num in range(1):#(orig_data_siz_z.shape[0]):
            subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])
            subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])
            test_image = imts[subject_id_start_slice:subject_id_end_slice,:,:]  
            subject_name = str(name_test_subjects[sub_num])[2:-1]
        
            # ================================================================
            # Extract features from the last layer (before the logits) of a TD image
            # ================================================================
            feed_dict={images_pl: np.expand_dims(test_image, axis=-1)}
            feats_last_layer = sess.run(features_last_layer, feed_dict=feed_dict)
            ptchs_last_layer = sess.run(patches_last_layer, feed_dict=feed_dict)
            random_indices = np.random.randint(0, ptchs_last_layer.shape[0], 100)

            # visualize feature maps 
            if (redraw == True) or (tf.gfile.Exists(log_dir_sd + prefix + 'features_td_' + subject_name + '.png') == False):
                utils_vis.save_features(feats_last_layer, savepath = log_dir_sd + prefix + 'features_' + subject_name + '.png')
            # visualize some patches
            if (redraw == True) or (tf.gfile.Exists(log_dir_sd + prefix + 'patches_td_' + subject_name + '.png') == False):
                utils_vis.save_patches(ptchs_last_layer,
                                    savepath = log_dir_sd + prefix + 'patches_td_' + subject_name + '.png',
                                    ids = random_indices,
                                    nc = 10,
                                    nr = 10,
                                    psize = args.patch_size)

            logging.info("Finding latent representation of TD patches:")
            td_patches_this_sub_latent = pca.transform(ptchs_last_layer)
            logging.info("size of latent representation of TD patches: " + str(td_patches_this_sub_latent.shape))

            # compute kdes for each latent dimension for this TD subject
            kdes_this_td_subject, z_vals = utils_kde.compute_pca_latent_kdes(td_patches_this_sub_latent, args.pca_kde_alpha)

            logging.info(kdes_this_td_subject.shape) # [50, 401]
            logging.info(kdes_all_sd_subjects.shape) # [15, 50, 401]

            # plot, for each latent dimension, the mean and std.deviation plots of KDEs over SD subjects + an overlay of this TD subject's KDE
            if (redraw == True) or (tf.gfile.Exists(log_dir_sd + prefix + 'kde_pca_coefs_td_' + subject_name + 'vs_sd_subjects.png') == False):
                utils_vis.plot_kdes_for_sd_and_td(kdes_this_td_subject,
                                                  kdes_all_sd_subjects,
                                                  z_vals,
                                                  savepath = log_dir_sd + prefix + 'kde_td_' + subject_name + 'vs_sd_subjects.png')
            

            # plot latent representations of SD and TD subjects
            # utils_vis.plot_scatter_pca_coefs_pairwise(ptchs_last_layer_td_latent,
            #                                           savepath = log_dir_sd + prefix + 'pca_coefs_td_' + subject_name + '_pairwise.png')
            
        # ================================================================
        # Close the session
        # ================================================================
        sess.close()
        
# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()
