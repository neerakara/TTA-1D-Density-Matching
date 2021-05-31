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
parser.add_argument('--test_dataset', default = "PROMISE") # PROMISE / USZ / CALTECH / STANFORD / HCPT2
parser.add_argument('--test_sub_num', type = int, default = 0) # 0 to 19
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
# Batch settings
parser.add_argument('--b_size', type = int, default = 16) # 1 / 2 / 4 (requires 24G GPU)
parser.add_argument('--feature_subsampling_factor', type = int, default = 8) # 1 / 4
parser.add_argument('--features_randomized', type = int, default = 1) # 1 / 0
# Matching settings
parser.add_argument('--match_with_sd', type = int, default = 2) # 1 / 2 / 3 / 4
# Learning rate settings
parser.add_argument('--tta_learning_rate', type = float, default = 0.001) # 0.001 / 0.0005 / 0.0001 
parser.add_argument('--tta_learning_sch', type = int, default = 1) # 0 / 1
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
        images_pl = tf.placeholder(tf.float32, shape = [args.b_size] + list(image_size) + [1], name = 'images')
    else:
        # Set first entry of shape to None to compute SD stats over entire volumes
        images_pl = tf.placeholder(tf.float32, shape = [args.b_size] + list(image_size) + [1], name = 'images')
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

                # Reshape to bring all those axes together where you want to take moments across
                features = tf.reshape(features, (-1, features.shape[-1]))

                # Subsample the feature maps to relieve the memory constraint and enable higher batch sizes
                if args.feature_subsampling_factor != 1:
                    
                    if args.features_randomized == 0:
                        features = features[::args.feature_subsampling_factor, :]
                    
                    elif args.features_randomized == 1:
                        # https://stackoverflow.com/questions/49734747/how-would-i-randomly-sample-pixels-in-tensorflow
                        # https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/gather.md
                        random_indices = tf.random.uniform(shape=[features.shape[0].value // args.feature_subsampling_factor],
                                                           minval=0,
                                                           maxval=features.shape[0].value - 1,
                                                           dtype=tf.int32)
                        features = tf.gather(features, random_indices, axis=0)

                # Compute first two moments of the computed features
                this_layer_means, this_layer_variances = tf.nn.moments(features, axes = [0])

                td_means = tf.concat([td_means, this_layer_means], 0)
                td_variances = tf.concat([td_variances, this_layer_variances], 0)

        td_mu = td_means[1:]
        td_var = td_variances[1:]

        # =================================
        # Compute the TTA loss - match Gaussians with KL loss
        # =================================
        loss_gaussian_kl_op = tf.reduce_mean(tf.math.log(td_var / sd_var_pl) + (sd_var_pl + (sd_mu_pl - td_mu)**2) / td_var)
        loss_op = loss_gaussian_kl_op # mean over all channels of all layers

        # ================================================================
        # add losses to tensorboard
        # ================================================================      
        tf.summary.scalar('loss/TTA', loss_op)         
        tf.summary.scalar('loss/Gaussian_KL', loss_gaussian_kl_op)
        summary_during_tta = tf.summary.merge_all()

    # ================================================================
    # Gaussian / FULL matching WITH KDEs
    # ================================================================
    elif args.KDE == 1:

        # placeholder for SD PDFs (mean over all SD subjects). These will be extracted after loading the SD trained model.
        # The shapes have to be hard-coded. Can't get the tile operations to work otherwise..
        sd_pdf_pl = tf.placeholder(tf.float32, shape = [704, 61], name = 'sd_pdfs') # shape [num_channels, num_points_along_intensity_range]
        # placeholder for the points at which the PDFs are evaluated
        x_pdf_pl = tf.placeholder(tf.float32, shape = [61], name = 'x_pdfs') # shape [num_points_along_intensity_range]
        # placeholder for the smoothing factor in the KDE computation
        alpha_pl = tf.placeholder(tf.float32, shape = [], name = 'alpha') # shape [1]

        # ================================================================
        # compute the pdfs of features of the TD image that is fed via the placeholder
        # ================================================================
        td_pdfs = tf.zeros([1, sd_pdf_pl.shape[1]]) # shape [num_channels, num_points_along_intensity_range]
        for conv_block in [1,2,3,4,5,6,7]:
            for conv_sub_block in [1,2]:
                conv_string = str(conv_block) + '_' + str(conv_sub_block)
                features_td = tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '_bn/FusedBatchNorm:0')
                features_td = tf.reshape(features_td, (-1, features_td.shape[-1]))

                # for Batch size 2:
                # 1_1 (131072, 16), 1_2 (131072, 16), 2_1 (32768, 32), 2_2 (32768, 32)
                # 3_1 (8192, 64), 3_2 (8192, 64), 4_1 (2048, 128), 4_2 (2048, 128)
                # 5_1 (8192, 64), 5_2 (8192, 64), 6_1 (32768, 32), 6_2 (32768, 32)
                # 7_1 (131072, 16), 7_2 (131072, 16)

                # Subsample the feature maps to relieve the memory constraint and enable higher batch sizes
                if args.feature_subsampling_factor != 1:
                    if args.features_randomized == 0:
                        features_td = features_td[::args.feature_subsampling_factor, :]
                    elif args.features_randomized == 1:
                        # https://stackoverflow.com/questions/49734747/how-would-i-randomly-sample-pixels-in-tensorflow
                        # https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/gather.md
                        random_indices = tf.random.uniform(shape=[features_td.shape[0].value // args.feature_subsampling_factor],
                                                            minval=0,
                                                            maxval=features_td.shape[0].value - 1,
                                                            dtype=tf.int32)
                        features_td = tf.gather(features_td, random_indices, axis=0)

                features_td = tf.tile(tf.expand_dims(features_td, 0), multiples = [x_pdf_pl.shape[0], 1, 1])
                x_pdf_tmp = tf.tile(tf.expand_dims(tf.expand_dims(x_pdf_pl, -1), -1), multiples = [1, features_td.shape[1], features_td.shape[2]])

                # the 3 dimensions are : 
                # 1. the intensity values where the pdf is evaluated,
                # 2. all the features (the pixels along the 2 spatial dimensions as well as the batch dimension are considered 1D iid samples)
                # 3. the channels 
                channel_pdf_this_layer_td = tf.reduce_mean(tf.math.exp(-alpha_pl * tf.math.square(x_pdf_tmp - features_td)), axis=1)
                channel_pdf_this_layer_td = tf.transpose(channel_pdf_this_layer_td)
                # at the end, we get 1 pdf (evaluated at the intensity values in x_pdf_pl) per channel
                
                td_pdfs = tf.concat([td_pdfs, channel_pdf_this_layer_td], 0)
        
        # Ignore the zeroth column that was added at the start of the loop
        td_pdfs = td_pdfs[1:, :]

        # ================================================================
        # compute the TTA loss - add ops for all losses and select based on the argument
        # ================================================================

        # ==================================
        # Match all moments with KL loss
        # ==================================
        # D_KL (p_s, p_t) = \sum_{x} p_s(x) log( p_s(x) / p_t(x) )
        loss_all_kl_op = tf.reduce_mean(tf.reduce_sum(tf.math.multiply(sd_pdf_pl,
                                                                       tf.math.log(tf.math.divide(sd_pdf_pl,
                                                                                                  td_pdfs + 1e-5) + 1e-2)), axis = 1))

        # ==================================
        # Match first two moments with KL loss
        # ==================================
        # compute means (across spatial locations and the batch axis) from the PDFs : $ \mu = \sum_{i=xmin}^{xmax} x * p(x) $
        x_pdf_tiled = tf.tile(tf.expand_dims(x_pdf_pl, 0), multiples = [td_pdfs.shape[0], 1]) # [Nc, Nx]
        td_pdf_means = tf.reduce_sum(tf.math.multiply(td_pdfs, x_pdf_tiled), axis = 1) # [Nc]
        sd_pdf_means = tf.reduce_sum(tf.math.multiply(sd_pdf_pl, x_pdf_tiled), axis = 1) # [Nc]
        # compute variances (across spatial locations and the batch axis) from the PDFs, using the means computed above
        # $ \sigma^2 = \sum_{i=xmin}^{xmax} (x - \mu)^2 * p(x) $
        td_pdf_variances_tmp = tf.math.square(x_pdf_tiled - tf.tile(tf.expand_dims(td_pdf_means, 1), multiples = [1, x_pdf_tiled.shape[1]]))
        td_pdf_variances = tf.reduce_sum(tf.math.multiply(td_pdfs, td_pdf_variances_tmp), axis = 1) # [Nc]
        sd_pdf_variances_tmp = tf.math.square(x_pdf_tiled - tf.tile(tf.expand_dims(sd_pdf_means, 1), multiples = [1, x_pdf_tiled.shape[1]]))
        sd_pdf_variances = tf.reduce_sum(tf.math.multiply(sd_pdf_pl, sd_pdf_variances_tmp), axis = 1) # [Nc]
        # D_KL (N(\mu_s, \sigma_s), N(\mu_t, \sigma_t)) = log(\sigma_t**2 / \sigma_s**2) + (\sigma_s**2 + (\mu_s - \mu_t)**2) / (\sigma_t**2)
        loss_gaussian_kl_op = tf.reduce_mean(tf.math.log(td_pdf_variances / sd_pdf_variances) + (sd_pdf_variances + (sd_pdf_means - td_pdf_means)**2) / td_pdf_variances)

        # ==================================
        # Match Full PDFs by min. L2 distance between the corresponding Characteristic Functions (complex)
        # ==================================
        # compute CFs of the source and target domains
        td_cfs = tf.spectral.rfft(td_pdfs)
        sd_cfs = tf.spectral.rfft(sd_pdf_pl)
        loss_all_cf_l2_op = tf.reduce_mean(tf.math.abs(td_cfs - sd_cfs)) # mean over all channels of all layers and all frequencies
        
        # ==================================
        # Select loss to be minimized according to the arguments
        # ==================================
        # match full PDFs with KL loss
        if args.match_moments == 'All_KL': 
            loss_op = loss_all_kl_op
        # match Gaussian with KL loss
        elif args.match_moments == 'Gaussian_KL':
            loss_op = loss_gaussian_kl_op
        # min L2 distance between complex arrays (match CFs exactly)
        elif args.match_moments == 'CF_L2':
            loss_op = loss_all_cf_l2_op
                
        # ================================================================
        # add losses to tensorboard
        # ================================================================
        tf.summary.scalar('loss/TTA', loss_op)         
        tf.summary.scalar('loss/All_KL', loss_all_kl_op)
        tf.summary.scalar('loss/Gaussian_KL', loss_gaussian_kl_op)
        tf.summary.scalar('loss/All_CF_L2', loss_all_cf_l2_op)
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
    display_pdfs_pl = tf.placeholder(tf.uint8, shape = [1, None, None, 1], name='display_pdfs_pl')
    pdfs_summary = tf.summary.image('PDFs', display_pdfs_pl)
    display_cfs_abs_pl = tf.placeholder(tf.uint8, shape = [1, None, None, 1], name='display_cfs_abs_pl')
    cfs_abs_summary = tf.summary.image('CFs_Magnitude', display_cfs_abs_pl)
    display_cfs_angle_pl = tf.placeholder(tf.uint8, shape = [1, None, None, 1], name='display_cfs_angle_pl')
    cfs_angle_summary = tf.summary.image('CFs_Phase', display_cfs_angle_pl)

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
    # compute the SD PDFs once (extract the whole pdf instead of just the 1st and 2nd moments of the pdf)
    # These will be passed as placeholders for computing the loss in each iteration
    # ================================================================
    if args.KDE == 1:
        b_size_compute_sd_pdfs = 2
        alpha = args.alpha
        res = 0.1
        x_min = -3.0
        x_max = 3.0
        pdf_str = 'alpha' + str(alpha) + 'xmin' + str(x_min) + 'xmax' + str(x_max) + '_res' + str(res) + '_bsize' + str(b_size_compute_sd_pdfs)
        x_values = np.arange(x_min, x_max + res, res)
        sd_pdfs_filename = path_to_model + 'sd_pdfs_' + pdf_str + '_subjectwise.npy'
        
        if os.path.isfile(sd_pdfs_filename):            
            pdfs_sd = np.load(sd_pdfs_filename) # [num_subjects, num_channels, num_x_points]
        
        else:
            pdfs_sd = []
            num_training_subjects = orig_data_siz_z_train.shape[0]            
            for train_sub_num in range(num_training_subjects):            
                
                logging.info("==== Computing pdf for subject " + str(train_sub_num) + '..')
                sd_image = imtr[np.sum(orig_data_siz_z_train[:train_sub_num]) : np.sum(orig_data_siz_z_train[:train_sub_num+1]),:,:]
                logging.info(sd_image.shape)
                
                num_batches = 0
                
                for b_i in range(0, sd_image.shape[0], b_size_compute_sd_pdfs):
                    if b_i + b_size_compute_sd_pdfs < sd_image.shape[0]: # ignoring the rest of the image (that doesn't fit the last batch) for now.
                        pdfs_this_batch = sess.run(td_pdfs, feed_dict={images_pl: np.expand_dims(sd_image[b_i:b_i+b_size_compute_sd_pdfs, ...], axis=-1),
                                                                        x_pdf_pl: x_values,
                                                                        alpha_pl: alpha})
                        if b_i == 0:
                            pdfs_this_subject = pdfs_this_batch

                        else:
                            pdfs_this_subject = pdfs_this_subject + pdfs_this_batch

                        num_batches = num_batches + 1
                
                pdfs_this_subject = pdfs_this_subject / num_batches

                pdfs_sd.append(pdfs_this_subject)

            pdfs_sd = np.array(pdfs_sd)

            # ================================================================
            # save
            # ================================================================
            np.save(sd_pdfs_filename, pdfs_sd) # [num_subjects, num_channels, num_x_points]

    # ================================================================
    # compute the SD Gaussian PDFs once
    # These will be passed as placeholders for computing the loss in each iteration
    # ================================================================
    elif args.KDE == 0:

        sd_gaussians_filename = path_to_model + 'sd_gaussians_' + args.before_or_after_bn + '_BN_subjectwise.npy'
        
        if os.path.isfile(sd_gaussians_filename):            
            gaussians_sd = np.load(sd_gaussians_filename) # [num_subjects, num_channels, 2]
        
        else:
            gaussians_sd = []
            num_training_subjects = orig_data_siz_z_train.shape[0]            
            for train_sub_num in range(num_training_subjects):            
                
                logging.info("==== Computing Gaussian for subject " + str(train_sub_num) + '..')
                sd_image = imtr[np.sum(orig_data_siz_z_train[:train_sub_num]) : np.sum(orig_data_siz_z_train[:train_sub_num+1]),:,:]
                logging.info(sd_image.shape)
                
                batchwise = False
                b_size_compute_sd_gaussians = 2
                # =========================
                # =========================
                if batchwise == True:
                    num_batches = 0
                    for b_i in range(0, sd_image.shape[0], b_size_compute_sd_gaussians):
                        if b_i + b_size_compute_sd_gaussians < sd_image.shape[0]: # ignoring the rest of the image (that doesn't fit the last batch) for now.                    
                            b_mu, b_var = sess.run([td_mu, td_var], feed_dict={images_pl: np.expand_dims(sd_image[b_i:b_i+b_size_compute_sd_gaussians, ...], axis=-1)})
                            if b_i == 0:
                                s_mu = b_mu
                                s_var = b_var
                            else:
                                s_mu = s_mu + b_mu
                                s_var = s_var + b_var
                            num_batches = num_batches + 1
                    s_mu = s_mu / num_batches
                    s_var = s_var / num_batches
                # =========================
                # =========================
                elif batchwise == False:
                    s_mu, s_var = sess.run([td_mu, td_var], feed_dict={images_pl: np.expand_dims(sd_image, axis=-1)})

                logging.info(s_mu.shape)
                logging.info(s_var.shape)

                # =========================
                # =========================             
                gaussians_sd.append(np.stack((s_mu, s_var), 1),)

            gaussians_sd = np.array(gaussians_sd)

            # ================================================================
            # save
            # ================================================================
            np.save(sd_gaussians_filename, gaussians_sd) # [num_subjects, num_channels, 2]
    
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

        # =============================
        # SD PDF / Gaussian to match with
        # =============================
        if args.match_with_sd == 1: # match with mean PDF over SD subjects
            if args.KDE == 1:
                sd_pdf_this_step = np.mean(pdfs_sd, axis = 0)
            else:
                sd_gaussian_this_step = np.mean(gaussians_sd, axis=0)

        elif args.match_with_sd == 2: # select a different SD subject for each TTA iteration
            if args.KDE == 1:
                sd_pdf_this_step = pdfs_sd[np.random.randint(pdfs_sd.shape[0]), :, :]
            else:
                sd_gaussian_this_step = gaussians_sd[np.random.randint(gaussians_sd.shape[0]), :, :]
                        
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
                           sd_pdf_pl: sd_pdf_this_step, 
                           x_pdf_pl: x_values, 
                           alpha_pl: alpha,
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
                batch = np.expand_dims(np.concatenate((test_image[b_i:, ...], np.zeros((extra_zeros_needed, test_image.shape[1], test_image.shape[2]))), axis=0), axis=-1)
            
            label_predicted.append(sess.run(preds, feed_dict={images_pl: batch}))
            image_normalized.append(sess.run(images_normalized, feed_dict={images_pl: batch}))

        label_predicted = np.squeeze(np.array(label_predicted)).astype(float)  
        image_normalized = np.squeeze(np.array(image_normalized)).astype(float)  

        if b_size > 1:
            label_predicted = np.reshape(label_predicted, (label_predicted.shape[0]*label_predicted.shape[1], label_predicted.shape[2], label_predicted.shape[3]))
            image_normalized = np.reshape(image_normalized, (image_normalized.shape[0]*image_normalized.shape[1], image_normalized.shape[2], image_normalized.shape[3]))
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
        if (step+1) % tta_vis_freq == 0:
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
            # visualize feature distribution alignment
            # ===========================
            if args.KDE == 1:
                b_size = args.b_size
                num_batches = 0
                for b_i in range(0, test_image.shape[0], b_size):
                    if b_i + b_size < test_image.shape[0]: # ignoring the rest of the image (that doesn't fit the last batch) for now.
                        pdfs_this_batch = sess.run(td_pdfs, feed_dict={images_pl: np.expand_dims(test_image[b_i:b_i+b_size, ...], axis=-1),
                                                                        x_pdf_pl: x_values,
                                                                        alpha_pl: alpha})
                        if b_i == 0:
                            pdfs_td_this_step = pdfs_this_batch
                        else:
                            pdfs_td_this_step = pdfs_td_this_step + pdfs_this_batch
                        num_batches = num_batches + 1
                pdfs_td_this_step = pdfs_td_this_step / num_batches

                utils_vis.write_pdfs(step,
                                    summary_writer,
                                    sess,
                                    pdfs_summary,
                                    display_pdfs_pl,
                                    np.mean(pdfs_sd, axis = 0),
                                    np.std(pdfs_sd, axis = 0),
                                    pdfs_td_this_step,
                                    x_values,
                                    log_dir_tta)

                # ===========================
                # visualize feature distribution alignment
                # ===========================
                b_i = 0
                sd_cfs_batch_this_step = sess.run(sd_cfs, feed_dict={sd_pdf_pl: np.mean(pdfs_sd, axis = 0)})
                td_cfs_batch_this_step = sess.run(td_cfs, feed_dict={images_pl: np.expand_dims(test_image[b_i:b_i+b_size, ...], axis=-1),
                                                                    x_pdf_pl: x_values,
                                                                    alpha_pl: alpha})

                utils_vis.write_cfs(step,
                                    summary_writer,
                                    sess,
                                    cfs_abs_summary,
                                    cfs_angle_summary,
                                    display_cfs_abs_pl,
                                    display_cfs_angle_pl,
                                    sd_cfs_batch_this_step,
                                    td_cfs_batch_this_step,
                                    log_dir_tta)

        step = step + 1

    # ================================================================
    # close session
    # ================================================================
    sess.close()