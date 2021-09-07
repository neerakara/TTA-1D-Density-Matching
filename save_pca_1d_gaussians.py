# ==================================================================
# import 
# ==================================================================
import logging
import os.path
import numpy as np
import tensorflow as tf
import argparse
import pickle as pk
import sklearn.metrics as met
from skimage.transform import rescale
from sklearn.decomposition import PCA

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
parser.add_argument('--train_dataset', default = "CSF") # RUNMC / CSF / UMC / HCPT1
parser.add_argument('--tr_run_number', type = int, default = 1) # 1 / 
parser.add_argument('--tr_cv_fold_num', type = int, default = 1) # 1 / 2

# PCA settings
parser.add_argument('--PCA_PSIZE', type = int, default = 16) # 16
parser.add_argument('--PCA_STRIDE', type = int, default = 8) # 8 (for all except UMC, where this needs to set to 2 to get enough 'fg' patches for all subjects)
parser.add_argument('--PCA_LAYER', default = 'layer_7_2') # layer_7_2 / logits / softmax
parser.add_argument('--PCA_LATENT_DIM', type = int, default = 10) # 10 / 50
parser.add_argument('--PCA_THRESHOLD', type = float, default = 0.8) # 0.8

parser.add_argument('--PDF_TYPE', default = "GAUSSIAN")

# visualization
parser.add_argument('--redraw', type = int, default = 0) # 0 / 1

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

# ================================================================
# Setup directories for this run
# ================================================================
if args.train_dataset == 'UMC':
    expname_i2l = 'tr' + args.train_dataset + '_cv' + str(args.tr_cv_fold_num) + '_r' + str(args.tr_run_number) + '/' + 'i2i2l/'
else:
    expname_i2l = 'tr' + args.train_dataset + '_r' + str(args.tr_run_number) + '/' + 'i2i2l/'
log_dir_sd = sys_config.project_root + 'log_dir/' + expname_i2l

# ==================================================================
# ==================================================================
def main():

    # ===================================
    # load training images
    # ===================================
    loaded_training_data = utils_data.load_training_data(args.train_dataset,
                                                         image_size,
                                                         target_resolution)
    imtr = loaded_training_data[0]
    orig_data_siz_z_train = loaded_training_data[7]
        
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
        
        # Get the last layer features. From here, there is a 1x1 conv that gives the logits
        features_last_layer = tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + str(7) + '_' + str(2) + '_bn/FusedBatchNorm:0')
        
        # Extract patches of the features of a certain channel of these features
        patches_last_layer = utils_kde.extract_patches(features_last_layer,
                                                       pca_channel_pl,
                                                       args.PCA_PSIZE,
                                                       args.PCA_STRIDE)

        # Accessing logits directly gives an error, but accesing it like this is fine! Weird TF!
        features_logits = tf.identity(logits)
        patches_logits = utils_kde.extract_patches(features_logits,
                                                   pca_channel_pl,
                                                   args.PCA_PSIZE,
                                                   args.PCA_STRIDE)

        # Get softmax and use these to select active patches
        features_softmax = tf.identity(softmax)
        patches_softmax = utils_kde.extract_patches(features_softmax,
                                                    channel = 2, # This needs to be 1 or 2 for prostate data
                                                    psize = args.PCA_PSIZE,
                                                    stride = args.PCA_STRIDE)

        # Combine the softmax scores into a map of foreground probabilities and use this to select active patches
        features_fg_probs = tf.expand_dims(tf.math.reduce_max(softmax[:, :, :, 1:], axis=-1), axis=-1)
        patches_fg_probs = utils_kde.extract_patches(features_fg_probs,
                                                     channel = 0, 
                                                     psize = args.PCA_PSIZE,
                                                     stride = args.PCA_STRIDE)        

        # ================================================================
        # divide the vars into segmentation network and normalization network
        # ================================================================
        i2l_vars = []
        for v in tf.global_variables():
            var_name = v.name        
            i2l_vars.append(v)
                                
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
        pca_dir = log_dir_sd + 'onedpdfs/' + exp_config.make_pca_dir_name(args)
        logging.info('PCA directory: ' + pca_dir)
        pca_figures_dir = pca_dir + 'figures/'
        if not tf.gfile.Exists(pca_dir):
            tf.gfile.MakeDirs(pca_dir)
        if not tf.gfile.Exists(pca_figures_dir):
            tf.gfile.MakeDirs(pca_figures_dir)

        # ================================================================
        # Do Channel-wise PCA
        # ================================================================
        if args.PCA_LAYER == 'layer_7_2':
            num_channels = 16
        else: # 'logits' / 'softmax'
            num_channels = nlabels

        # ====================
        # go through all channels of this feature layer
        # ====================
        for channel in range(num_channels):

            logging.info("==================================")
            logging.info("Channel " + str(channel))

            pca_filepath = pca_dir + 'c' + str(channel) + '.pkl'
            if os.path.exists(pca_filepath):
                logging.info("PCA already done. Loading saved model..")
                pca = pk.load(open(pca_filepath, 'rb'))
                num_pcs = pca.n_components_
            else:
                # ====================
                # init arrays to store features and patches of different subjects
                # ====================
                if args.train_dataset == 'HCPT1':
                    num_subjects = 10 # takes quite long to do this for all 20 subjects. Also 10 subjects should likely provide enough variability.
                else:
                    num_subjects = orig_data_siz_z_train.shape[0]
                sd_features = np.zeros([num_subjects, image_size[0], image_size[1]]) 
                sd_patches = np.zeros([1, args.PCA_PSIZE*args.PCA_PSIZE])
                sd_patches_active = np.zeros([1, args.PCA_PSIZE*args.PCA_PSIZE])

                # ====================
                # go through all sd subjects
                # ====================
                for train_sub_num in range(num_subjects):

                    # ====================
                    # Extract the image
                    # ====================
                    train_image = imtr[np.sum(orig_data_siz_z_train[:train_sub_num]) : np.sum(orig_data_siz_z_train[:train_sub_num+1]),:,:]
                    
                    # ====================
                    # extract features and patches
                    # ====================
                    feed_dict={images_pl: np.expand_dims(train_image, axis=-1), pca_channel_pl: channel}
                    if args.PCA_LAYER == 'layer_7_2':
                        feats_last_layer = sess.run(features_last_layer, feed_dict=feed_dict)
                        ptchs_last_layer = sess.run(patches_last_layer, feed_dict=feed_dict)
                    elif args.PCA_LAYER == 'logits':
                        feats_last_layer = sess.run(features_logits, feed_dict=feed_dict)
                        ptchs_last_layer = sess.run(patches_logits, feed_dict=feed_dict)
                    elif args.PCA_LAYER == 'softmax':
                        feats_last_layer = sess.run(features_softmax, feed_dict=feed_dict)
                        ptchs_last_layer = sess.run(patches_softmax, feed_dict=feed_dict)
                    
                    # ====================
                    # get corresponding patches of the foreground probability values
                    # These will be used to select active patches for this subject
                    # ====================
                    ptchs_fg_probs = sess.run(patches_fg_probs, feed_dict=feed_dict)

                    # ====================
                    # collect features from the central slice for vis
                    # ====================
                    sd_features[train_sub_num, :, :] = feats_last_layer[feats_last_layer.shape[0]//2, :, :, channel]

                    # ====================
                    # number of patches from this subject
                    # ====================
                    logging.info("Number of patches in SD subject " + str(train_sub_num+1) + ": " + str(ptchs_last_layer.shape[0]))
                    sd_patches = np.concatenate((sd_patches, ptchs_last_layer), axis=0)

                    # ====================
                    # extract 'active' patches -> ones for which the softmax of class two has a high value in the central pixel
                    # ====================
                    active_ptchs_last_layer = ptchs_last_layer[np.where(ptchs_fg_probs[:, (args.PCA_PSIZE * (args.PCA_PSIZE + 1))//2] > args.PCA_THRESHOLD)[0], :]
                    logging.info("Number of active patches in SD subject " + str(train_sub_num+1) + ": " + str(active_ptchs_last_layer.shape[0]))
                    sd_patches_active = np.concatenate((sd_patches_active, active_ptchs_last_layer), axis=0)

                # ====================
                # remove dummy patch added in the front, before the loop over all sd subjects
                # ====================
                sd_patches = np.array(sd_patches[1:,:])
                sd_patches_active = np.array(sd_patches_active[1:,:])
                logging.info("Number of all SD patches:" + str(sd_patches.shape[0]))
                logging.info("Number of all active SD patches:" + str(sd_patches_active.shape[0]))

                # ====================
                # visualize feature maps 
                # ====================
                if (args.redraw == 1) or (tf.gfile.Exists(pca_figures_dir + 'SD_features_c' + str(channel) + '.png') == False):
                    utils_vis.save_features(sd_features, savepath = pca_figures_dir + 'SD_features_c' + str(channel) + '.png')
                
                # ====================
                # visualize some patches (randomly selected from all SD subjects)
                # ====================
                if (args.redraw == 1) or (tf.gfile.Exists(pca_figures_dir + 'SD_patches_c' + str(channel) + '.png') == False):
                    random_indices = np.random.randint(0, sd_patches.shape[0], 100)
                    utils_vis.save_patches(sd_patches,
                                        savepath = pca_figures_dir + 'SD_patches_c' + str(channel) + '.png',
                                        ids = random_indices,
                                        nc = 5,
                                        nr = 5,
                                        psize = args.PCA_PSIZE)

                    random_indices_active = np.random.randint(0, sd_patches_active.shape[0], 100)
                    utils_vis.save_patches(sd_patches_active,
                                        savepath = pca_figures_dir + 'SD_active_patches_c' + str(channel) + '.png',
                                        ids = random_indices_active,
                                        nc = 5,
                                        nr = 5,
                                        psize = args.PCA_PSIZE)

                # ============================================================
                # PCA
                # ============================================================
                num_pcs = args.PCA_LATENT_DIM
                pca = PCA(n_components = num_pcs, whiten=True)
                logging.info("Fitting PCA parameters to centered SD patches from all subjects")
                pca.fit(sd_patches_active)
                # The fit method subtracts the mean of each feature and saves the subtracted mean
                # The mean is then used while transforming new datapoints from x to z or z to x.
                # write to file
                pk.dump(pca, open(pca_filepath, "wb"))

            # ====================
            # how much variance is explained by the first k components
            # ====================
            logging.info("sum of ratios of variance explained by the first " + str(num_pcs) + " PCs:" + str(np.round(np.sum(pca.explained_variance_ratio_), 2)))

            # ====================
            # visualize the principal components
            # ====================
            if (args.redraw == 1) or (tf.gfile.Exists(pca_figures_dir + 'pcs_c' + str(channel) + '.png') == False):
                logging.info("Visualizing principal components..")
                utils_vis.visualize_principal_components(pca.components_, pca_figures_dir + 'pcs_c' + str(channel) + '.png', args.PCA_PSIZE, nc = 2, nr = 2)

            # ====================
            # Compute KDEs of each latent dimension for patches of each SD training subject      
            # ====================
            logging.info("Computing Gaussians in each latent dimension for individual SD training subjects..")
            gaussians_all_sd_tr_subs = utils_kde.compute_latent_gaussians_subjectwise(images = imtr,
                                                                                      train_dataset = args.train_dataset,
                                                                                      image_size = image_size,
                                                                                      image_depths = orig_data_siz_z_train,
                                                                                      image_placeholder = images_pl,
                                                                                      channel_placeholder = pca_channel_pl,
                                                                                      channel_num = channel,
                                                                                      features = features_last_layer,
                                                                                      patches = patches_last_layer,
                                                                                      fg_probs = patches_fg_probs,
                                                                                      psize = args.PCA_PSIZE,
                                                                                      threshold = args.PCA_THRESHOLD,
                                                                                      learned_pca = pca,
                                                                                      sess = sess,
                                                                                      savepath = pca_dir + 'c' + str(channel) + '.npy')

        # ================================================================
        # Close the session
        # ================================================================
        sess.close()
        
# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()
