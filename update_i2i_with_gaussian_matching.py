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
import sklearn.metrics as met
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
parser.add_argument('--train_dataset', default = "NCI") # NCI / HCPT1
parser.add_argument('--tr_run_number', type = int, default = 1) # 1 / 
# Test dataset and subject number
parser.add_argument('--test_dataset', default = "USZ") # PROMISE / USZ / CALTECH / STANFORD / HCPT2
parser.add_argument('--test_sub_num', type = int, default = 0) # 0 to 19
# TTA options
parser.add_argument('--tta_string', default = "TTA/")
parser.add_argument('--adaBN', type = int, default = 0) # 0 / 1
# Whether to compute KDE or not?
parser.add_argument('--KDE', type = int, default = 1) # 0 / 1
parser.add_argument('--alpha', type = float, default = 100.0) # 10.0 / 100.0 / 1000.0
parser.add_argument('--KDE_Groups', type = int, default = 1) # 0 / 1
# Which vars to adapt?
parser.add_argument('--tta_vars', default = "NORM") # BN / NORM
# How many moments to match and how?
parser.add_argument('--match_moments', default = "All_KL") # Gaussian_KL / All_KL / All_CF_L2
parser.add_argument('--before_or_after_bn', default = "AFTER") # AFTER / BEFORE
# Batch settings
parser.add_argument('--b_size', type = int, default = 16) # 1 / 2 / 4 (requires 24G GPU)
parser.add_argument('--feature_subsampling_factor', type = int, default = 16) # 1 / 8
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
tta_max_steps = dataset_params[6]
tta_model_saving_freq = dataset_params[7]
tta_vis_freq = dataset_params[8]
b_size_compute_sd_pdfs = dataset_params[9]
b_size_compute_sd_gaussians = dataset_params[10]

# ================================================================
# load training data (for computing SD PDFs)
# ================================================================
imtr, gttr, orig_data_siz_z_train, num_train_subjects = utils.load_training_data(args.train_dataset,
                                                                                 image_size,
                                                                                 target_resolution)

# ================================================================
# load test data
# ================================================================
loaded_test_data = utils.load_testing_data(args.test_dataset,
                                           image_size,
                                           target_resolution,
                                           image_depth_ts)

imts = loaded_test_data[0]
gtts = loaded_test_data[1]
orig_data_res_z = loaded_test_data[4]
orig_data_siz_z = loaded_test_data[7]
name_test_subjects = loaded_test_data[8]
num_test_subjects = loaded_test_data[9]

# ================================================================
# Make the name for this TTA run
# ================================================================
exp_str = exp_config.make_tta_exp_name(args)

# ================================================================
# Extract test image (TTA for the asked subject) / Set test_ids for SFDA for the requested TD
# ================================================================
if args.TTA_or_SFDA == 'TTA':
    sub_num = args.test_sub_num    
    logging.info(str(name_test_subjects[sub_num])[2:-1])
    subject_name = str(name_test_subjects[sub_num])[2:-1]
    subject_string = args.test_dataset + '_' + subject_name
    exp_str = exp_str + subject_string

    # extract the single test volume
    subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])
    subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])
    test_image = imts[subject_id_start_slice:subject_id_end_slice,:,:]  
    test_image_gt = gtts[subject_id_start_slice:subject_id_end_slice,:,:]  
    test_image_gt = test_image_gt.astype(np.uint8)

elif args.TTA_or_SFDA == 'SFDA':
    if args.test_dataset == 'USZ':
        td_string = 'SFDA_' + args.test_dataset
        test_ids = np.arange(num_test_subjects)

    elif args.test_dataset == 'PROMISE':
        td_string = 'SFDA_' + args.test_dataset + '_' + args.PROMISE_SUB_DATASET
        if args.PROMISE_SUB_DATASET == 'RUNMC':
            test_ids = np.array([15, 4, 6, 18, 14, 3]) # cases 11, 14, 16, 19, 21, 24
        elif args.PROMISE_SUB_DATASET == 'UCL':
            test_ids = np.array([11, 7, 1, 5, 16, 9]) # cases 1, 26, 29, 31, 34, 36
        elif args.PROMISE_SUB_DATASET == 'BIDMC':
            test_ids = np.array([8, 2, 19]) # cases 4, 6, 9
        elif args.PROMISE_SUB_DATASET == 'HK':
            test_ids = np.array([10, 13, 17, 12, 0]) # 39, 41, 44, 46, 49

    exp_str = exp_str + td_string

# ================================================================
# Setup directories for this run
# ================================================================
expname_i2l = 'tr' + args.train_dataset + '_r' + str(args.tr_run_number) + '/' + 'i2i2l/'
log_dir = sys_config.project_root + 'log_dir/' + expname_i2l
log_dir_tta = log_dir + exp_str
tensorboard_dir_tta = sys_config.tensorboard_root + expname_i2l + exp_str
logging.info('SD training directory: %s' %log_dir)
logging.info('Tensorboard directory TTA: %s' %tensorboard_dir_tta)
if not tf.gfile.Exists(log_dir_tta):
    tf.gfile.MakeDirs(log_dir_tta)
    tf.gfile.MakeDirs(tensorboard_dir_tta)

# ================================================================
# build the TF graph
# ================================================================
with tf.Graph().as_default():
    
    # ================================================================
    # create placeholders
    # ================================================================
    if args.KDE == 1:
        # If SD stats have not been computed so far, run once with b_size set to b_size_compute_sd_pdfs
        images_pl = tf.placeholder(tf.float32, shape = [args.b_size] + list(image_size) + [1], name = 'images')
    else:
        # If SD stats have not been computed so far, run once with b_size set to b_size_compute_sd_gaussians
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

    # ================================================================
    # Set TTA vars
    # ================================================================
    if args.tta_vars == "BN":
        tta_vars = bn_vars
    elif args.tta_vars == "NORM":
        tta_vars = normalization_vars

    # ================================================================
    # Gaussian matching without computing KDE
    # ================================================================
    if args.KDE == 0:

        # placeholders for SD stats. These will be extracted after loading the SD trained model.
        sd_mu_pl = tf.placeholder(tf.float32, shape = [None], name = 'sd_means')
        sd_var_pl = tf.placeholder(tf.float32, shape = [None], name = 'sd_variances')

        # compute the stats of features of the TD image that is fed via the placeholder
        td_means = tf.zeros([1])
        td_variances = tf.ones([1])
        for conv_block in [1,2,3,4,5,6,7]:
            for conv_sub_block in [1,2]:
                conv_string = str(conv_block) + '_' + str(conv_sub_block)

                # Whether to compute Gaussians before or after BN layers
                if args.before_or_after_bn == 'BEFORE':
                    features = tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '/Conv2D:0')
                elif args.before_or_after_bn == 'AFTER':
                    features = tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '_bn/FusedBatchNorm:0')

                # Compute the first two moments for all channels of this layer
                this_layer_means, this_layer_variances = utils_kde.compute_first_two_moments(features,
                                                                                             args.feature_subsampling_factor,
                                                                                             args.features_randomized)
                td_means = tf.concat([td_means, this_layer_means], 0)
                td_variances = tf.concat([td_variances, this_layer_variances], 0)

        # Also add logits to the features where priors are computed
        if args.use_logits_for_TTA == 1:
            features = tf.get_default_graph().get_tensor_by_name('i2l_mapper/pred/Conv2D:0')
            this_layer_means, this_layer_variances = utils_kde.compute_first_two_moments(features,
                                                                                         args.feature_subsampling_factor,
                                                                                         args.features_randomized)
            td_means = tf.concat([td_means, this_layer_means], 0)
            td_variances = tf.concat([td_variances, this_layer_variances], 0)

        # Remove the 'dummy' first entry
        td_mu = td_means[1:]
        td_var = td_variances[1:]

        # =================================
        # Compute KL divergence between Gaussians
        # =================================
        loss_gaussian_kl_op = tf.reduce_mean(tf.math.log(td_var / sd_var_pl) + (sd_var_pl + (sd_mu_pl - td_mu)**2) / td_var)

        # =================================
        # Total loss
        # =================================
        loss_op = loss_gaussian_kl_op # mean over all channels of all layers

        # ================================================================
        # Add losses to tensorboard
        # ================================================================      
        tf.summary.scalar('loss/TTA', loss_op)         
        tf.summary.scalar('loss/Gaussian_KL', loss_gaussian_kl_op)
        summary_during_tta = tf.summary.merge_all()

    # ================================================================
    # Gaussian / FULL matching WITH KDEs
    # ================================================================
    elif args.KDE == 1:

        # =======================================================
        # DIVIDE CHANNELS into 3 groups:
            # 1. 0 to 688 (layers 1_2 to 7_1) [range of channel means: [-0.5, 0.5], range of channel variances: [0.25, 2.5]]
            # 2. 689 to 704 (layer 7_2) [range of channel means: [-0.5, 1.5], range of channel variances: [2.5, 8.0]]
            # 3. 704 to 704 + n_labels (logits) [range of channel means: [-15, 20], range of channel variances: [15, 100]]
        # KDEs of channels in groups 2 and 3 need to be computed to an extended range of values.
        # =======================================================

        # placeholder for SD PDFs (mean over all SD subjects). These will be extracted after loading the SD trained model.
        # The shapes have to be hard-coded. Can't get the tile operations to work otherwise..
        sd_pdf_g1_pl = tf.placeholder(tf.float32, shape = [688, 61], name = 'sd_pdfs_g1') # shape [num_channels, num_points_along_intensity_range]
        sd_pdf_g2_pl = tf.placeholder(tf.float32, shape = [16, 101], name = 'sd_pdfs_g2') # shape [num_channels, num_points_along_intensity_range]
        sd_pdf_g3_pl = tf.placeholder(tf.float32, shape = [nlabels, 601], name = 'sd_pdfs_g3') # shape [num_channels, num_points_along_intensity_range]
        # placeholder for the points at which the PDFs are evaluated
        x_pdf_g1_pl = tf.placeholder(tf.float32, shape = [61], name = 'x_pdfs_g1') # shape [num_points_along_intensity_range]
        x_pdf_g2_pl = tf.placeholder(tf.float32, shape = [101], name = 'x_pdfs_g2') # shape [num_points_along_intensity_range]
        x_pdf_g3_pl = tf.placeholder(tf.float32, shape = [601], name = 'x_pdfs_g3') # shape [num_points_along_intensity_range]
        # placeholder for the smoothing factor in the KDE computation
        alpha_pl = tf.placeholder(tf.float32, shape = [], name = 'alpha') # shape [1]

        # =======================================================
        # compute the pdfs of features of the TD image that is fed via the placeholder
        # =======================================================

        # ==================================
        # GROUP 1
        # ==================================
        td_pdfs_g1 = tf.zeros([1, sd_pdf_g1_pl.shape[1]]) # shape [num_channels, num_points_along_intensity_range]
        for conv_block in [1,2,3,4,5,6,7]:
            for conv_sub_block in [1,2]:
                if conv_block == 7 and conv_sub_block == 2:
                    continue
                conv_string = str(conv_block) + '_' + str(conv_sub_block)
                features_td = tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '_bn/FusedBatchNorm:0')
                channel_pdf_this_layer_td = utils_kde.compute_feature_kdes(features_td,
                                                                           args.feature_subsampling_factor,
                                                                           args.features_randomized,
                                                                           x_pdf_g1_pl,
                                                                           alpha_pl)
                td_pdfs_g1 = tf.concat([td_pdfs_g1, channel_pdf_this_layer_td], 0)
        # Ignore the zeroth column that was added at the start of the loop
        td_pdfs_g1 = td_pdfs_g1[1:, :]

        # ==================================
        # GROUP 2 (layer 7_2)
        # ==================================
        features_td = tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + str(7) + '_' + str(2) + '_bn/FusedBatchNorm:0')
        td_pdfs_g2 = utils_kde.compute_feature_kdes(features_td,
                                                    args.feature_subsampling_factor,
                                                    args.features_randomized,
                                                    x_pdf_g2_pl,
                                                    alpha_pl)

        # ==================================
        # GROUP 3 (logits)
        # ==================================
        features_td = tf.get_default_graph().get_tensor_by_name('i2l_mapper/pred/Conv2D:0')
        td_pdfs_g3 = utils_kde.compute_feature_kdes(features_td,
                                                    args.feature_subsampling_factor,
                                                    args.features_randomized,
                                                    x_pdf_g3_pl,
                                                    alpha_pl)

        # ================================================================
        # compute the TTA loss - add ops for all losses and select based on the argument
        # ================================================================
        loss_all_kl_g1_op = utils_kde.compute_kl_between_kdes(sd_pdf_g1_pl, td_pdfs_g1)
        loss_all_kl_g2_op = utils_kde.compute_kl_between_kdes(sd_pdf_g2_pl, td_pdfs_g2)
        loss_all_kl_g3_op = utils_kde.compute_kl_between_kdes(sd_pdf_g3_pl, td_pdfs_g3)
        
        # ==================================
        # Select loss to be minimized according to the arguments
        # ==================================
        # match full PDFs with KL loss
        if args.match_moments == 'All_KL': 
            loss_op = loss_all_kl_g1_op + loss_all_kl_g2_op + loss_all_kl_g3_op
                
        # ================================================================
        # add losses to tensorboard
        # ================================================================
        tf.summary.scalar('loss/TTA', loss_op)         
        tf.summary.scalar('loss/All_KL_G1', loss_all_kl_g1_op)
        tf.summary.scalar('loss/All_KL_G2', loss_all_kl_g2_op)
        tf.summary.scalar('loss/All_KL_G3', loss_all_kl_g3_op)
        summary_during_tta = tf.summary.merge_all()
    
    # ================================================================
    # add optimization ops
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
    # ================================================================                        
    loss_ema_pl = tf.placeholder(tf.float32, shape = [], name = 'loss_ema') # shape [1]
    loss_ema_summary = tf.summary.scalar('loss/TTA_EMA', loss_ema_pl)

    # ================================================================
    # add init ops
    # ================================================================
    init_ops = tf.global_variables_initializer()
    init_tta_ops = tf.initialize_variables(tta_vars) # set TTA vars to random values
            
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
    saver_tta = tf.train.Saver(var_list = tta_vars, max_to_keep=10)   
    saver_tta_best = tf.train.Saver(var_list = tta_vars, max_to_keep=3)   
            
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
    # compute the SD Gaussian PDFs once
    # These will be passed as placeholders for computing the loss in each iteration
    # ================================================================
    if args.KDE == 0:
        
        sd_gaussians_fname = exp_config.make_sd_gaussian_names(path_to_model,
                                                               b_size_compute_sd_gaussians,
                                                               args)

        gaussians_sd = utils_kde.compute_sd_gaussians(sd_gaussians_fname,
                                                      args.train_dataset,
                                                      imtr,
                                                      image_depth_tr,
                                                      orig_data_siz_z_train,
                                                      b_size_compute_sd_gaussians,
                                                      sess,
                                                      td_mu,
                                                      td_var,
                                                      images_pl)

    # ================================================================
    # compute the SD PDFs once (extract the whole pdf instead of just the 1st and 2nd moments of the pdf)
    # These will be passed as placeholders for computing the loss in each iteration
    # ================================================================
    elif args.KDE == 1:
        kde_g1_params = [-3.0, 3.0, 0.1]
        x_values_g1 = np.arange(kde_g1_params[0], kde_g1_params[1] + kde_g1_params[2], kde_g1_params[2])
        sd_pdfs_fname_g1 = exp_config.make_sd_pdf_name(path_to_model, b_size_compute_sd_pdfs, args, 1, kde_g1_params)
        # [num_subjects, num_channels, num_x_points]
        pdfs_sd_g1 = utils_kde.compute_sd_pdfs(sd_pdfs_fname_g1,
                                               args.train_dataset,
                                               imtr,
                                               image_depth_tr,
                                               orig_data_siz_z_train,
                                               b_size_compute_sd_pdfs,
                                               sess,
                                               td_pdfs_g1,
                                               images_pl,
                                               x_pdf_g1_pl,
                                               x_values_g1,
                                               alpha_pl,
                                               args.alpha)

        kde_g2_params = [-5.0, 5.0, 0.1]
        x_values_g2 = np.arange(kde_g2_params[0], kde_g2_params[1] + kde_g2_params[2], kde_g2_params[2])
        sd_pdfs_fname_g2 = exp_config.make_sd_pdf_name(path_to_model, b_size_compute_sd_pdfs, args, 2, kde_g2_params)
        # [num_subjects, num_channels, num_x_points]
        pdfs_sd_g2 = utils_kde.compute_sd_pdfs(sd_pdfs_fname_g2,
                                               args.train_dataset,
                                               imtr,
                                               image_depth_tr,
                                               orig_data_siz_z_train,
                                               b_size_compute_sd_pdfs,
                                               sess,
                                               td_pdfs_g2,
                                               images_pl,
                                               x_pdf_g2_pl,
                                               x_values_g2,
                                               alpha_pl,
                                               args.alpha)

        kde_g3_params = [-30.0, 30.0, 0.1]
        x_values_g3 = np.arange(kde_g3_params[0], kde_g3_params[1] + kde_g3_params[2], kde_g3_params[2])
        sd_pdfs_fname_g3 = exp_config.make_sd_pdf_name(path_to_model, b_size_compute_sd_pdfs, args, 3, kde_g3_params)
        # [num_subjects, num_channels, num_x_points]
        pdfs_sd_g3 = utils_kde.compute_sd_pdfs(sd_pdfs_fname_g3,
                                               args.train_dataset,
                                               imtr,
                                               image_depth_tr,
                                               orig_data_siz_z_train,
                                               b_size_compute_sd_pdfs,
                                               sess,
                                               td_pdfs_g3,
                                               images_pl,
                                               x_pdf_g3_pl,
                                               x_values_g3,
                                               alpha_pl,
                                               args.alpha)

    # ===================================
    # Set TTA vars to random values at the start of TTA, if requested
    # ===================================
    if args.tta_init_from_scratch == 1:
        sess.run(init_tta_ops)

    # ===================================
    # TTA / SFDA iterations
    # ===================================
    step = 0
    best_loss = 1000.0
    if args.TTA_or_SFDA == 'SFDA':
        tta_max_steps = num_test_subjects * tta_max_steps

    while (step < tta_max_steps):
        
        logging.info("TTA / SFDA step: " + str(step+1))
        
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
        if args.match_with_sd == 1: # match with mean PDF over SD subjects
            if args.KDE == 1:
                sd_pdf_g1_this_step = np.mean(pdfs_sd_g1, axis = 0)
                sd_pdf_g2_this_step = np.mean(pdfs_sd_g2, axis = 0)
                sd_pdf_g3_this_step = np.mean(pdfs_sd_g3, axis = 0)
            else:
                sd_gaussian_this_step = np.mean(gaussians_sd, axis=0)

        elif args.match_with_sd == 2: # select a different SD subject for each TTA iteration
            if args.KDE == 1:              
                sub_id = np.random.randint(pdfs_sd_g1.shape[0])  
                sd_pdf_g1_this_step = pdfs_sd_g1[sub_id, :, :]
                sd_pdf_g2_this_step = pdfs_sd_g2[sub_id, :, :]
                sd_pdf_g3_this_step = pdfs_sd_g3[sub_id, :, :]
            else:
                sub_id = np.random.randint(gaussians_sd.shape[0])
                sd_gaussian_this_step = gaussians_sd[sub_id, :, :]
                        
        # =============================
        # For SFDA, select a different TD subject in each adaptation epochs
        # =============================
        if args.TTA_or_SFDA == 'SFDA':
            sub_num = test_ids[np.random.randint(test_ids.shape[0])]
            subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])
            subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])
            test_image = imts[subject_id_start_slice:subject_id_end_slice,:,:]  
            test_image_gt = gtts[subject_id_start_slice:subject_id_end_slice,:,:]  
            test_image_gt = test_image_gt.astype(np.uint8)

        # =============================
        # Adaptation iterations within this epoch
        # =============================
        b_size = args.b_size
        for b_i in range(0, test_image.shape[0], b_size):
            if args.KDE == 1:      
                feed_dict={images_pl: np.expand_dims(test_image[np.random.randint(0, test_image.shape[0], b_size), :, :], axis=-1),
                           sd_pdf_g1_pl: sd_pdf_g1_this_step, 
                           sd_pdf_g2_pl: sd_pdf_g2_this_step, 
                           sd_pdf_g3_pl: sd_pdf_g3_this_step, 
                           x_pdf_g1_pl: x_values_g1, 
                           x_pdf_g2_pl: x_values_g2, 
                           x_pdf_g3_pl: x_values_g3, 
                           alpha_pl: args.alpha,
                           lr_pl: tta_learning_rate}

            elif args.KDE == 0:      
                feed_dict={images_pl: np.expand_dims(test_image[np.random.randint(0, test_image.shape[0], b_size), :, :], axis=-1),
                           sd_mu_pl: sd_gaussian_this_step[:,0], 
                           sd_var_pl: sd_gaussian_this_step[:,1],
                           lr_pl: tta_learning_rate}

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

        if b_size > 1:
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

        if args.test_dataset == 'PROMISE':
            label_predicted[label_predicted!=0] = 1
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
                                            test_image_gt)

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
                logging.info(sd_image.shape)
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
            if args.KDE == 0:

                b_size = args.b_size
                num_batches = 0
                for b_i in range(0, test_image.shape[0], b_size):
                    if b_i + b_size < test_image.shape[0]: # ignoring the rest of the image (that doesn't fit the last batch) for now.                    
                        b_mu, b_var = sess.run([td_mu, td_var], feed_dict={images_pl: np.expand_dims(test_image[b_i:b_i+b_size, ...], axis=-1)})
                        if b_i == 0:
                            test_mu = b_mu
                            test_var = b_var
                        else:
                            test_mu = test_mu + b_mu
                            test_var = test_var + b_var
                        num_batches = num_batches + 1
                test_mu = test_mu / num_batches
                test_var = test_var / num_batches

                utils_vis.write_gaussians(step,
                                          summary_writer,
                                          sess,
                                          pdfs_summary,
                                          display_pdfs_pl,
                                          np.mean(gaussians_sd, axis=0)[:,0],
                                          np.mean(gaussians_sd, axis=0)[:,1],
                                          test_mu,
                                          test_var,
                                          log_dir_tta,
                                          args.use_logits_for_TTA,
                                          nlabels)

            elif args.KDE == 1:
                b_size = args.b_size
                num_batches = 0
                for b_i in range(0, test_image.shape[0], b_size):
                    
                    if b_i + b_size < test_image.shape[0]: # ignoring the rest of the image (that doesn't fit the last batch) for now.

                        batch_tmp = np.expand_dims(test_image[b_i:b_i+b_size, ...], axis=-1)
                        pdfs_g1_this_batch = sess.run(td_pdfs_g1, feed_dict={images_pl: batch_tmp, x_pdf_g1_pl: x_values_g1, alpha_pl: args.alpha})
                        pdfs_g2_this_batch = sess.run(td_pdfs_g2, feed_dict={images_pl: batch_tmp, x_pdf_g2_pl: x_values_g2, alpha_pl: args.alpha})
                        pdfs_g3_this_batch = sess.run(td_pdfs_g3, feed_dict={images_pl: batch_tmp, x_pdf_g3_pl: x_values_g3, alpha_pl: args.alpha})
                        if b_i == 0:
                            pdfs_td_g1_this_step = pdfs_g1_this_batch
                            pdfs_td_g2_this_step = pdfs_g2_this_batch
                            pdfs_td_g3_this_step = pdfs_g3_this_batch
                        else:
                            pdfs_td_g1_this_step = pdfs_td_g1_this_step + pdfs_g1_this_batch
                            pdfs_td_g2_this_step = pdfs_td_g2_this_step + pdfs_g2_this_batch
                            pdfs_td_g3_this_step = pdfs_td_g3_this_step + pdfs_g3_this_batch
                        num_batches = num_batches + 1

                pdfs_td_g1_this_step = pdfs_td_g1_this_step / num_batches
                pdfs_td_g2_this_step = pdfs_td_g2_this_step / num_batches
                pdfs_td_g3_this_step = pdfs_td_g3_this_step / num_batches

                utils_vis.write_pdfs(step,
                                     summary_writer,
                                     sess,
                                     pdfs_summary,
                                     display_pdfs_pl,
                                     np.mean(pdfs_sd_g1, axis = 0), np.std(pdfs_sd_g1, axis = 0), pdfs_td_g1_this_step, x_values_g1,
                                     np.mean(pdfs_sd_g2, axis = 0), np.std(pdfs_sd_g2, axis = 0), pdfs_td_g2_this_step, x_values_g2,
                                     np.mean(pdfs_sd_g3, axis = 0), np.std(pdfs_sd_g3, axis = 0), pdfs_td_g3_this_step, x_values_g3,
                                     log_dir_tta)

        step = step + 1

    # ================================================================
    # close session
    # ================================================================
    sess.close()