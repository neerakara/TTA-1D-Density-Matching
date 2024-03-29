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
from tfwrapper import layers

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
# Whether to compute KDE or not?
parser.add_argument('--KDE', type = int, default = 0) # 0 to 1
parser.add_argument('--alpha', type = float, default = 100.0) # 10.0 / 100.0 / 1000.0
# Which vars to adapt?
parser.add_argument('--tta_vars', default = "NORM") # BN / NORM
# How many moments to match and how?
parser.add_argument('--match_moments', default = "Gaussian_KL") # Gaussian_KL / All_KL / All_CF_L2
parser.add_argument('--before_or_after_bn', default = "AFTER") # AFTER / BEFORE
# MRF settings
parser.add_argument('--BINARY', default = 0) # 1 / 0
parser.add_argument('--POTENTIAL_TYPE', type = int, default = 3) # 1 / 2
parser.add_argument('--BINARY_LAMBDA', type = float, default = 0.1) # 1.0
parser.add_argument('--BINARY_ALPHA', type = float, default = 1.0) # 1.0 / 10.0 (smoothness paramter for the KDE of the binary potentials)
# Random Feature settings
parser.add_argument('--RANDOM_FEATURES', default = 1) # 1 / 0
parser.add_argument('--PATCH_SIZE', type = int, default = 100) # 20
parser.add_argument('--NUM_RANDOM_FEATURES', type = int, default = 50) # 50
# If modeling the distribution of the random features with a Gaussian distribution
parser.add_argument('--RANDOM_FEATURES_GAUSSIAN_COV', default = "DIAG") # DIAG / FULL
# If modeling the distribution of the random features with 1 1D KDE per dimension
parser.add_argument('--RANDOM_FEATURES_KDE_ALPHA', type = float, default = 1.0) # 1.0 / 10.0 (smoothness paramter for the KDE of the random features)
# Batch settings
parser.add_argument('--b_size', type = int, default = 16) # 1 / 2 / 4 (requires 24G GPU)
parser.add_argument('--feature_subsampling_factor', type = int, default = 1) # 1 / 8
parser.add_argument('--features_randomized', type = int, default = 0) # 1 / 0
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
    # Get features of one of the last layers and reduce their dimensionality with a random projection
    # ================================================================
    # last layer. From here, there is a 1x1 conv that gives the logits
    conv_string = str(7) + '_' + str(2)
    features_last_layer = tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '_bn/FusedBatchNorm:0')
    logging.info("Feature size in the layer 7_2: " + str(features_last_layer.shape))
    num_patches = 25*(256//args.PATCH_SIZE)*(256//args.PATCH_SIZE)
    features_last_layer_rp = layers.conv2D_layer(features_last_layer,
                                                 'random_projection',
                                                 kernel_size=args.PATCH_SIZE,
                                                 num_filters=args.NUM_RANDOM_FEATURES,
                                                 strides=args.PATCH_SIZE,
                                                 padding="valid")
    logging.info("Feature size after random projection: " + str(features_last_layer_rp.shape))
    logging.info("Orig dimension of patches: " + str(features_last_layer_rp.shape[-1]*args.PATCH_SIZE*args.PATCH_SIZE))
    logging.info("Number of patches: " + str(num_patches))
    logging.info("Reduced representation size: " + str(args.NUM_RANDOM_FEATURES))
    logging.info("E-value according to Johnson Linderstrauss Theorem: " + str(np.sqrt(np.log(num_patches)/args.NUM_RANDOM_FEATURES)))

    # ================================================================
    # divide the vars into segmentation network and normalization network
    # ================================================================
    i2l_vars = []
    normalization_vars = []
    bn_vars = []
    rp_vars = []
    for v in tf.global_variables():
        var_name = v.name       
        if 'random_projection' in var_name:
            rp_vars.append(v)
        else:
            i2l_vars.append(v)
            if 'image_normalizer' in var_name:
                normalization_vars.append(v)
            if 'beta' in var_name or 'gamma' in var_name:
                bn_vars.append(v)

    # logging.info('------------- random projection vars')
    # for v in rp_vars: logging.info(v.name)
    # logging.info('------------- i2l vars')
    # for v in i2l_vars: logging.info(v.name)
    # logging.info('------------- normalization vars')
    # for v in normalization_vars: logging.info(v.name)
    # logging.info('------------- bn vars')
    # for v in bn_vars: logging.info(v.name)

    # ================================================================
    # Set TTA vars
    # ================================================================
    if args.tta_vars == "BN":
        tta_vars = bn_vars
    elif args.tta_vars == "NORM":
        tta_vars = normalization_vars

    # ================================================================
    # Gaussian matching of RANDOM FEATURES
    # ================================================================
    if args.KDE == 0:

        # placeholders for SD stats. These will be extracted after loading the SD trained model.
        sd_mu_pl = tf.placeholder(tf.float32, shape = [args.NUM_RANDOM_FEATURES], name = 'sd_means')
        if args.RANDOM_FEATURES_GAUSSIAN_COV == 'DIAG':
            sd_var_pl = tf.placeholder(tf.float32, shape = [args.NUM_RANDOM_FEATURES], name = 'sd_variances')
        elif args.RANDOM_FEATURES_GAUSSIAN_COV == 'FULL':
            sd_var_pl = tf.placeholder(tf.float32, shape = [args.NUM_RANDOM_FEATURES, args.NUM_RANDOM_FEATURES], name = 'sd_variances')

        # Compute the first two moments for each dimension of the random projection
        # Diagonal Covariance across the different dimensions for now..
        td_mu, td_var = utils_kde.compute_first_two_moments(features_last_layer_rp,
                                                            args.feature_subsampling_factor,
                                                            args.features_randomized,
                                                            args.RANDOM_FEATURES_GAUSSIAN_COV)

        # =================================
        # Compute KL divergence between Gaussians
        # =================================
        loss_gaussian_kl_op = tf.reduce_mean(tf.math.log(td_var / sd_var_pl) + (sd_var_pl + (sd_mu_pl - td_mu)**2) / td_var)
        loss_op = loss_gaussian_kl_op 

        # ================================================================
        # Add losses to tensorboard
        # ================================================================      
        tf.summary.scalar('loss/TTA', loss_op)         
        tf.summary.scalar('loss/Gaussian_KL', loss_gaussian_kl_op)
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
    # create savers
    # ================================================================
    saver_i2l = tf.train.Saver(var_list = i2l_vars)
    saver_rp = tf.train.Saver(var_list = rp_vars)
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
        
        sd_gaussians_fname = exp_config.make_sd_RP_gaussian_names(path_to_model,
                                                                  b_size_compute_sd_gaussians,
                                                                  args)

        # If the SD distributions have been computed, load the RP matrices used for those computations
        if os.path.isfile(sd_gaussians_fname):            
            saver_rp.restore(sess, sd_gaussians_fname[:-4] + '_rp_matrices.ckpt')
        # else, save the random projection matrices that will be used to compute the SD distributions
        else:
            saver_rp.save(sess, sd_gaussians_fname[:-4] + '_rp_matrices.ckpt')

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
            sd_gaussian_this_step = np.mean(gaussians_sd, axis=0)
        elif args.match_with_sd == 2: # select a different SD subject for each TTA iteration
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
            
            feed_dict={images_pl: np.expand_dims(test_image[np.random.randint(0, test_image.shape[0], b_size), :, :], axis=-1),
                       sd_mu_pl: sd_gaussian_this_step[:,0], 
                       sd_var_pl: sd_gaussian_this_step[:,1],
                       lr_pl: tta_learning_rate}                

            # if args.KDE == 0:
            #     logging.info(np.round(sd_gaussian_this_step[:5,0], 2))
            #     logging.info(np.round(sd_gaussian_this_step[:5,1], 2))

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
        if step % tta_vis_freq == 0:

            # ===========================
            # I. Visualize image, normalized image, predictions and ground truth segmentations
            # ===========================
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
            # II. Visualize features of layer 7_2 for the test image and a randomly chosen SD image
            # ===========================
            display_features = 1
            if display_features == 1:

                # get Test image featuers
                tmp = test_image.shape[0] // 2 - b_size//2
                features_for_display_td = sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv7_2_bn/FusedBatchNorm:0'),
                                                   feed_dict={images_pl: np.expand_dims(test_image[tmp:tmp+b_size, ...], axis=-1)})

                # get SD image featuers
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
            # III. Visualize alignment between gaussian distributions of filter responses
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
                                          np.mean(gaussians_sd, axis=0)[:,0], # mean over all SD subjects of (means of 1d Gaussians for each dimension)
                                          np.mean(gaussians_sd, axis=0)[:,1], # mean over all SD subjects of (vars of 1d Gaussians for each dimension)
                                          test_mu,
                                          test_var,
                                          log_dir_tta,
                                          args.use_logits_for_TTA,
                                          nlabels,
                                          deltas = [0, 4, 8, 12],
                                          num_channels = 4)

        step = step + 1

    # ================================================================
    # close session
    # ================================================================
    sess.close()