# ==================================================================
# The following things happen in this file:
# ==================================================================
# 1. Get dataset dependent params (image_size, resolution, etc)
# 2. Load test data
# 3. Set paths and directories for the requested test ID
# 4. Extract test image for the requested test TD
# 5. Build the TF graph (normalization, segmentation and autoencoder networks)
# 6. Define loss function for AE loss minimization
# 7. Define optimization routine (gradient aggregation over all batches to cover the image volume)
# 8. Define summary ops
# 9. Define savers
# 10. TTA iterations
# 11. Visualizations:
# a. Image, normalized image and predicted labels
# b. Features of SD vs TD
# ==================================================================

# ==================================================================
# import 
# ==================================================================
import logging
import os.path
import argparse
import numpy as np
import pickle as pk
import tensorflow as tf
import sklearn.metrics as met

import utils
import utils_vis
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
parser.add_argument('--train_dataset', default = "RUNMC") # RUNMC (prostate) | CSF (cardiac) | UMC (brain white matter hyperintensities) | HCPT1 (brain subcortical tissues) | site2
parser.add_argument('--tr_run_number', type = int, default = 1) # 1 / 
parser.add_argument('--tr_cv_fold_num', type = int, default = 1) # 1 / 2
# Test dataset and subject number
parser.add_argument('--test_dataset', default = "HK") # BMC / USZ / UCL / BIDMC / HK (prostate) | UHE / HVHD (cardiac) | UMC / NUHS (brain WMH) | CALTECH (brain tissues) | site3
parser.add_argument('--test_cv_fold_num', type = int, default = 1) # 1 / 2
parser.add_argument('--test_sub_num', type = int, default = 0) # 0 to 19

# TTA base string
parser.add_argument('--tta_string', default = "tta/")
parser.add_argument('--tta_method', default = "AE")
parser.add_argument('--ae_runnum', type = int, default = 1) # 1 / 2 
# Which vars to adapt?
parser.add_argument('--TTA_VARS', default = "NORM") # BN / NORM / AdaptAx / AdaptAxAf
# which AEs
parser.add_argument('--whichAEs', default = "xn_f1_f2_f3_y") # xn / xn_y / xn_f1_f2_f3_y

# Batch settings
parser.add_argument('--b_size', type = int, default = 8)

# Learning rate settings
parser.add_argument('--tta_learning_rate', type = float, default = 1e-4) # 0.001 / 0.0005 / 0.0001 
parser.add_argument('--tta_learning_sch', type = int, default = 0) # 0 / 1
parser.add_argument('--tta_runnum', type = int, default = 1) # 1 / 2 / 3

# Which vars to adapt?
parser.add_argument('--accum_gradients', type = int, default = 1) # 0 / 1

# weight of spectral norm loss compared to the AE recon loss
parser.add_argument('--lambda_spectral', type = float, default = 1.0) # 1.0 / 5.0

# whether to print debug stuff or not
parser.add_argument('--debug', type = int, default = 0) # 1 / 0

# whether to train Ax first or not
parser.add_argument('--train_Ax_first', type = int, default = 0) # 1 / 0
parser.add_argument('--instance_norm_in_Ax', type = int, default = 0) # 1 / 0

# number channels in features that are autoencoded
parser.add_argument('--num_channels_f1', type = int, default = 32) # 16 / 32
parser.add_argument('--num_channels_f2', type = int, default = 64) # 32 / 64
parser.add_argument('--num_channels_f3', type = int, default = 128) # 64 / 128

# parse arguments
args = parser.parse_args()

# ================================================================
# set dataset dependent hyperparameters
# ================================================================
dataset_params = exp_config.get_dataset_dependent_params(args.train_dataset, args.test_dataset) 
image_size = dataset_params[0]
nlabels = dataset_params[1]
target_resolution = dataset_params[2]
image_depth_ts = dataset_params[4]
tta_max_steps = dataset_params[6]
tta_model_saving_freq = dataset_params[7]
tta_vis_freq = dataset_params[8]

# ================================================================
# load test data
# ================================================================
loaded_test_data = utils_data.load_testing_data(args.test_dataset,
                                                args.test_cv_fold_num,
                                                image_size,
                                                target_resolution,
                                                image_depth_ts)

imts = loaded_test_data[0]
gtts = loaded_test_data[1]
orig_data_res_x = loaded_test_data[2]
orig_data_res_y = loaded_test_data[3]
orig_data_res_z = loaded_test_data[4]
orig_data_siz_x = loaded_test_data[5]
orig_data_siz_y = loaded_test_data[6]
orig_data_siz_z = loaded_test_data[7]
name_test_subjects = loaded_test_data[8]
num_test_subjects = loaded_test_data[9]

# ================================================================
# Set paths and directories for the requested test ID
# ================================================================
sub_num = args.test_sub_num    
subject_name = str(name_test_subjects[sub_num])[2:-1]
logging.info(subject_name)

# dir where the SD mdoels have been saved
expname_i2l = 'tr' + args.train_dataset + '_cv' + str(args.tr_cv_fold_num) + '_r' + str(args.tr_run_number) + '/' + 'i2i2l/'
log_dir = sys_config.project_root + 'log_dir/' + expname_i2l

# dir for TTA
exp_str = exp_config.make_tta_exp_name(args, tta_method = args.tta_method) + args.test_dataset + '_' + subject_name
log_dir_tta = log_dir + exp_str
tensorboard_dir_tta = sys_config.tensorboard_root + expname_i2l + exp_str

logging.info('SD training directory: %s' %log_dir)
logging.info('Tensorboard directory TTA: %s' %tensorboard_dir_tta)

if not tf.gfile.Exists(log_dir_tta):
    tf.gfile.MakeDirs(log_dir_tta)
    tf.gfile.MakeDirs(tensorboard_dir_tta)

# ================================================================
# Run if not TTA not done before
# ================================================================
if not tf.gfile.Exists(log_dir_tta + '/models/model.ckpt-999.index'):

    # ================================================================
    # Extract test image for the requested TD
    # ================================================================
    subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])
    subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])
    test_image = imts[subject_id_start_slice:subject_id_end_slice,:,:]  
    test_image_gt = gtts[subject_id_start_slice:subject_id_end_slice,:,:]  
    test_image_gt = test_image_gt.astype(np.uint8)

    # ================================================================
    # Build the TF graph
    # ================================================================
    with tf.Graph().as_default():
        
        # ============================
        # Set random seed for reproducibility
        # ============================
        tf.random.set_random_seed(args.tta_runnum)
        np.random.seed(args.tta_runnum)
        
        # ================================================================
        # Create placeholders
        # ================================================================
        images_pl = tf.placeholder(tf.float32, shape = [args.b_size] + list(image_size) + [1], name = 'images')
        # setting training flag to false (relevant for batch normalization layers)
        training_pl = tf.constant(False, dtype=tf.bool)

        if args.TTA_VARS in ['AdaptAxAf', 'AdaptAx']:

            # ================================================================
            # Insert a randomly initialized 1x1 'adaptor' even before the normalization module.
            # To follow the procedure used in He MedIA 2021, we will adapt this module for each test volume, and keep the normalization module fixed at the values learned on the SD.
            # ================================================================
            images_adapted = model.adapt_Ax(images_pl, exp_config, instance_norm = args.instance_norm_in_Ax)

            # ================================================================
            # Insert a normalization module in front of the segmentation network
            # the normalization module is adapted for each test image
            # ================================================================
            images_normalized, added_residual = model.normalize(images_adapted, exp_config, training_pl = tf.constant(False, dtype=tf.bool))

            # ================================================================
            # Build the graph that computes predictions from the inference model
            # ================================================================
            logits, softmax, preds, features_level1, features_level2, features_level3 = model.predict_i2l_with_adaptors(images_normalized,
                                                                                                                        exp_config,
                                                                                                                        training_pl = tf.constant(False, dtype=tf.bool),
                                                                                                                        nlabels = nlabels,
                                                                                                                        return_features = True)

        else: # Directly feed the input image to the normalization module
            # ================================================================
            # Insert a normalization module in front of the segmentation network
            # the normalization module is adapted for each test image
            # ================================================================ 
            images_normalized, added_residual = model.normalize(images_pl, exp_config, training_pl = tf.constant(False, dtype=tf.bool))

            # ================================================================
            # Build the graph that computes predictions from the inference model
            # ================================================================        
            logits, softmax, preds, features_level1, features_level2, features_level3 = model.predict_i2l(images_normalized,
                                                                                                          exp_config,
                                                                                                          training_pl = training_pl,
                                                                                                          nlabels = nlabels,
                                                                                                          return_features = True)

        # ======================
        # autoencoder on the space of normalized images and on the softmax outputs
        # ======================
        images_normalized_autoencoded = model.autoencode(images_normalized, exp_config, tf.constant(False, dtype=tf.bool), 'xn')
        softmax_autoencoded = model.autoencode(softmax, exp_config, tf.constant(False, dtype=tf.bool), 'y')
        features_level1_autoencoded = model.autoencode(features_level1, exp_config, tf.constant(False, dtype=tf.bool), 'f1')
        features_level2_autoencoded = model.autoencode(features_level2, exp_config, tf.constant(False, dtype=tf.bool), 'f2')
        features_level3_autoencoded = model.autoencode(features_level3, exp_config, tf.constant(False, dtype=tf.bool), 'f3')
        
        # ================================================================
        # Divide the vars into different groups
        # ================================================================
        i2l_vars, normalization_vars, bn_vars, adapt_ax_vars, adapt_af_vars, ae_xn_vars, ae_y_vars, ae_f1_vars, ae_f2_vars, ae_f3_vars = model.divide_vars_into_groups(tf.global_variables(), AEs = True)

        if args.debug == 1:
            logging.info("Ax vars")
            for v in adapt_ax_vars:
                logging.info(v.name)
                logging.info(v.shape)
            
            logging.info("Af vars")
            for v in adapt_af_vars:
                logging.info(v.name)
                logging.info(v.shape)

        # ================================================================
        # ops for initializing feature adaptors to identity
        # ================================================================
        if args.TTA_VARS in ['AdaptAxAf', 'AdaptAx']:
            wf1 = [v for v in tf.global_variables() if v.name == "i2l_mapper/adaptAf_A1/kernel:0"][0]
            wf2 = [v for v in tf.global_variables() if v.name == "i2l_mapper/adaptAf_A2/kernel:0"][0]
            wf3 = [v for v in tf.global_variables() if v.name == "i2l_mapper/adaptAf_A3/kernel:0"][0]
            if args.debug == 1:
                logging.info("Weight matrices of feature adaptors.. ")
                logging.info(wf1)
                logging.info(wf2)
                logging.info(wf3)
            wf1_init_pl = tf.placeholder(tf.float32, shape = [1,1,args.num_channels_f1,args.num_channels_f1], name = 'wf1_init_pl')
            wf2_init_pl = tf.placeholder(tf.float32, shape = [1,1,args.num_channels_f2,args.num_channels_f2], name = 'wf2_init_pl')
            wf3_init_pl = tf.placeholder(tf.float32, shape = [1,1,args.num_channels_f3,args.num_channels_f3], name = 'wf3_init_pl')
            init_wf1_op = wf1.assign(wf1_init_pl)
            init_wf2_op = wf2.assign(wf2_init_pl)
            init_wf3_op = wf3.assign(wf3_init_pl)

            # op for optimizing Ax to be near identity
            init_ax_op = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(tf.reduce_mean(tf.square(images_adapted - images_pl)), var_list=adapt_ax_vars)

        # ================================================================
        # Set TTA vars
        # ================================================================
        if args.TTA_VARS == "BN":
            tta_vars = bn_vars
        elif args.TTA_VARS == "NORM":
            tta_vars = normalization_vars
        elif args.TTA_VARS == "AdaptAx":
            tta_vars = adapt_ax_vars
        elif args.TTA_VARS == "AdaptAxAf":
            tta_vars = adapt_ax_vars + adapt_af_vars

        # ================================================================
        # TTA loss (recon error of the AE)
        # ================================================================
        loss_op_xn = tf.reduce_mean(tf.math.square(images_normalized_autoencoded - images_normalized))
        loss_op_y = tf.reduce_mean(tf.math.square(softmax_autoencoded - softmax))
        loss_op_f1 = tf.reduce_mean(tf.math.square(features_level1_autoencoded - features_level1))
        loss_op_f2 = tf.reduce_mean(tf.math.square(features_level2_autoencoded - features_level2))
        loss_op_f3 = tf.reduce_mean(tf.math.square(features_level3_autoencoded - features_level3))
        
        if args.whichAEs == 'xn':
            loss_ae_op = loss_op_xn
        elif args.whichAEs == 'xn_y':
            loss_ae_op = loss_op_xn + loss_op_y
        elif args.whichAEs == 'xn_f1_f2_f3_y':
            loss_ae_op = loss_op_xn + loss_op_y + loss_op_f1 + loss_op_f2 + loss_op_f3

        # ================================================================
        # TTA loss - spectral norm of the feature adaptors
        # ================================================================
        if args.TTA_VARS in ['AdaptAxAf', 'AdaptAx']:
            # wf1_ = tf.transpose(wf1[0,0,:,:]) * wf1[0,0,:,:] - tf.eye(args.num_channels_f1)
            # loss_spectral_norm_wf1_op = tf.linalg.svd(wf1_, compute_uv=False)[...,0]
            # wf2_ = tf.transpose(wf2[0,0,:,:]) * wf2[0,0,:,:] - tf.eye(args.num_channels_f2)
            # loss_spectral_norm_wf2_op = tf.linalg.svd(wf2_, compute_uv=False)[...,0]
            # wf3_ = tf.transpose(wf3[0,0,:,:]) * wf3[0,0,:,:] - tf.eye(args.num_channels_f3)
            # loss_spectral_norm_wf3_op = tf.linalg.svd(wf3_, compute_uv=False)[...,0]

            loss_spectral_norm_wf1_op = model.spectral_loss(wf1)
            loss_spectral_norm_wf2_op = model.spectral_loss(wf2)
            loss_spectral_norm_wf3_op = model.spectral_loss(wf3)
            loss_spectral_norm_op = loss_spectral_norm_wf1_op + loss_spectral_norm_wf2_op + loss_spectral_norm_wf3_op

            # ================================================================
            # Total TTA loss
            # ================================================================
            loss_op = loss_ae_op + args.lambda_spectral * loss_spectral_norm_op

        else:
            # ================================================================
            # Total TTA loss
            # ================================================================
            loss_op = loss_ae_op
                
        # ================================================================
        # add losses to tensorboard
        # ================================================================
        tf.summary.scalar('loss/TTA', loss_op)
        tf.summary.scalar('loss/TTA__AE_total', loss_ae_op)
        tf.summary.scalar('loss/TTA__XN_', loss_op_xn)
        tf.summary.scalar('loss/TTA__Y_', loss_op_y)
        tf.summary.scalar('loss/TTA__F1_', loss_op_f1)
        tf.summary.scalar('loss/TTA__F2_', loss_op_f2)
        tf.summary.scalar('loss/TTA__F3_', loss_op_f3)
        if args.TTA_VARS in ['AdaptAxAf', 'AdaptAx']:
            tf.summary.scalar('loss/TTA_spectral_norm', loss_spectral_norm_op)
        summary_during_tta = tf.summary.merge_all()

        # ================================================================
        # Add optimization ops
        # ================================================================           
        lr_pl = tf.placeholder(tf.float32, shape = [], name = 'tta_learning_rate') # shape [1]
        # create an instance of the required optimizer
        optimizer = exp_config.optimizer_handle(learning_rate = lr_pl)    
        
        if args.accum_gradients == 1:
            # initialize variable holding the accumlated gradients and create a zero-initialisation op
            accumulated_gradients = [tf.Variable(tf.zeros_like(var.initialized_value()), trainable=False) for var in tta_vars]
            # accumulated gradients init op
            accumulated_gradients_zero_op = [ac.assign(tf.zeros_like(ac)) for ac in accumulated_gradients]
            # calculate gradients and define accumulation op
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            logging.info("----------------------- UPDATE OPS")
            for op in update_ops: logging.info(op.name)
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

        elif args.accum_gradients == 0:
            train_op = optimizer.minimize(loss_op, var_list = tta_vars)
            opt_memory_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            train_op = tf.group([train_op, opt_memory_update_ops])

        # ================================================================
        # placeholder for logging a smoothened loss
        # ================================================================                        
        loss_whole_subject_pl = tf.placeholder(tf.float32, shape = [], name = 'loss_whole_subject') # shape [1]
        loss_whole_subject_summary = tf.summary.scalar('loss/TTA_whole_subject', loss_whole_subject_pl)
        loss_ema_pl = tf.placeholder(tf.float32, shape = [], name = 'loss_ema') # shape [1]
        loss_ema_summary = tf.summary.scalar('loss/TTA_EMA', loss_ema_pl)

        # ================================================================
        # add init ops
        # ================================================================
        init_ops = tf.global_variables_initializer()
                
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

        # ================================================================
        # create saver
        # ================================================================
        saver_i2l = tf.train.Saver(var_list = i2l_vars)
        # savers to restore ae models
        saver_ae_xn = tf.train.Saver(var_list = ae_xn_vars)
        saver_ae_y = tf.train.Saver(var_list = ae_y_vars)
        saver_ae_f1 = tf.train.Saver(var_list = ae_f1_vars)
        saver_ae_f2 = tf.train.Saver(var_list = ae_f2_vars)
        saver_ae_f3 = tf.train.Saver(var_list = ae_f3_vars)
        # tta savers (we need multiple of these to save according to different stopping criteria)
        saver_tta = tf.train.Saver(var_list = tta_vars, max_to_keep=1) # general saver after every few epochs
        saver_tta_init = tf.train.Saver(var_list = tta_vars, max_to_keep=1) # saves the initial values of the TTA vars
        saver_tta_best = tf.train.Saver(var_list = tta_vars, max_to_keep=1) # saves the weights when the exponential moving average of the loss is the lowest
        saver_tta_best_first_10_epochs = tf.train.Saver(var_list = tta_vars, max_to_keep=1) # like saver_tta_best, but restricted to the first 10 epochs 
        saver_tta_best_first_50_epochs = tf.train.Saver(var_list = tta_vars, max_to_keep=1) # like saver_tta_best, but restricted to the first 50 epochs 
        saver_tta_best_first_100_epochs = tf.train.Saver(var_list = tta_vars, max_to_keep=1) # like saver_tta_best, but restricted to the first 100 epochs 
        saver_tta_sos = tf.train.Saver(var_list = tta_vars, max_to_keep=1) # saves weights at the iteration when the loss increases as compared to the previous step
                
        # ================================================================
        # freeze the graph before execution
        # ================================================================
        tf.get_default_graph().finalize()

        # ================================================================
        # Run the Op to initialize the variables.
        # ================================================================
        sess.run(init_ops)

        if args.TTA_VARS in ['AdaptAxAf', 'AdaptAx']:
            sess.run(init_wf1_op, feed_dict={wf1_init_pl: np.expand_dims(np.expand_dims(np.eye(args.num_channels_f1), axis=0), axis=0)})
            sess.run(init_wf2_op, feed_dict={wf2_init_pl: np.expand_dims(np.expand_dims(np.eye(args.num_channels_f2), axis=0), axis=0)})
            sess.run(init_wf3_op, feed_dict={wf3_init_pl: np.expand_dims(np.expand_dims(np.eye(args.num_channels_f3), axis=0), axis=0)})

            if args.debug == 1:
                logging.info('Initialized feature adaptors..')   
                logging.info(wf1.eval(session=sess))
                logging.info(wf2.eval(session=sess))
                logging.info(wf3.eval(session=sess))

            if args.train_Ax_first == 1:
                logging.info('Training Ax to be the identity mapping..')
                for _ in range(100):
                    sess.run(init_ax_op, feed_dict={images_pl: np.expand_dims(test_image[np.random.randint(0, test_image.shape[0], args.b_size), :, :], axis=-1)})
                logging.info('Done.. now doing TTA ops from here..')
        
        # ================================================================
        # Restore the normalization + segmentation network parameters
        # ================================================================
        path_to_model = sys_config.project_root + 'log_dir/' + expname_i2l + 'models/'
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_dice.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_i2l.restore(sess, checkpoint_path)

        # ================================================================
        # Restore the autoencoder (Xn) parameters
        # ================================================================
        path_to_ae_models = sys_config.project_root + 'log_dir/' + expname_i2l + 'tta/AE/r' + str(args.ae_runnum) + '/models_'
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_ae_models + 'xn/', 'best_loss.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_ae_xn.restore(sess, checkpoint_path)

        # ================================================================
        # Restore the autoencoder (Y) parameters
        # ================================================================
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_ae_models + 'y/', 'best_loss.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_ae_y.restore(sess, checkpoint_path)

        # ================================================================
        # Restore the autoencoder (F1) parameters
        # ================================================================
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_ae_models + 'f1/', 'best_loss.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_ae_f1.restore(sess, checkpoint_path)

        # ================================================================
        # Restore the autoencoder (F2) parameters
        # ================================================================
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_ae_models + 'f2/', 'best_loss.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_ae_f2.restore(sess, checkpoint_path)

        # ================================================================
        # Restore the autoencoder (F3) parameters
        # ================================================================
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_ae_models + 'f3/', 'best_loss.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_ae_f3.restore(sess, checkpoint_path)

        # =============================
        # compute TTA loss before any updates are done
        # =============================
        loss_iter_count = 0
        loss_step_zero = 0.0
        b_size = args.b_size
        for b_i in range(0, test_image.shape[0], b_size):
            if (b_i+1)*b_size > test_image.shape[0]:
                continue
            loss_step_zero = loss_step_zero + sess.run(loss_op, feed_dict = {images_pl: np.expand_dims(test_image[b_i*b_size:(b_i+1)*b_size, :, :], axis=-1)})
            loss_iter_count += 1
        loss_step_zero = loss_step_zero / loss_iter_count
        best_loss = loss_step_zero
        loss_ema = best_loss
        loss_previous_step = best_loss

        # ===========================
        # compute dice wrt gt before any updates are done
        # ===========================
        label_predicted = []
        for b_i in range(0, test_image.shape[0], b_size):
            if b_i + b_size < test_image.shape[0]:
                batch = np.expand_dims(test_image[b_i:b_i+b_size, ...], axis=-1)
            else:
                # pad zeros to have complete batches
                extra_zeros_needed = b_i + b_size - test_image.shape[0]
                batch = np.expand_dims(np.concatenate((test_image[b_i:, ...], np.zeros((extra_zeros_needed, test_image.shape[1], test_image.shape[2]))), axis=0), axis=-1)
            label_predicted.append(sess.run(preds, feed_dict={images_pl: batch}))
        label_predicted = np.squeeze(np.array(label_predicted)).astype(float)  

        if b_size > 1 and test_image.shape[0] > b_size:
            label_predicted = np.reshape(label_predicted, (label_predicted.shape[0]*label_predicted.shape[1], label_predicted.shape[2], label_predicted.shape[3]))
        label_predicted = label_predicted[:test_image.shape[0], ...]
        if args.test_dataset in ['UCL', 'HK', 'BIDMC']:
            label_predicted[label_predicted!=0.0] = 1.0
        dice_wrt_gt = met.f1_score(test_image_gt.flatten(), label_predicted.flatten(), average=None) 
        
        # Record dice and TTA loss at step 0 (before adaptation)
        summary_writer.add_summary(sess.run(gt_dice_summary, feed_dict={gt_dice: np.mean(dice_wrt_gt[1:])}), 0)
        summary_writer.add_summary(sess.run(loss_whole_subject_summary, feed_dict={loss_whole_subject_pl: loss_step_zero}), 0)
        summary_writer.add_summary(sess.run(loss_ema_summary, feed_dict={loss_ema_pl: loss_ema}), 0)

        # This will be set to True once a SOS model is saved, so that this model is not overwritten
        model_sos_saved = False

        # save initial values of the TTA vars (useful to check how much does the performance degrade due to random init of the adaptor modules)
        saver_tta_init.save(sess, os.path.join(log_dir_tta, 'models/tta_init.ckpt'), global_step=0)

        # ===================================
        # TTA / SFDA iterations
        # ===================================
        step = 1

        while (step < tta_max_steps):
            
            logging.info("TTA step: " + str(step))
            
            # =============================
            # run accumulated_gradients_zero_op (no need to provide values for any placeholders)
            # =============================
            if args.accum_gradients == 1:    
                sess.run(accumulated_gradients_zero_op)
            num_accumulation_steps = 0

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
            # Adaptation iterations within this epoch
            # =============================
            for b_i in range(0, test_image.shape[0], b_size):

                feed_dict={images_pl: np.expand_dims(test_image[np.random.randint(0, test_image.shape[0], b_size), :, :], axis=-1), lr_pl: tta_learning_rate}
                
                # run the accumulate gradients op 
                if args.accum_gradients == 1:    
                    sess.run(accumulate_gradients_op, feed_dict=feed_dict)
                # run the train op on this batch
                elif args.accum_gradients == 0:    
                    sess.run(train_op, feed_dict=feed_dict)
                    
                num_accumulation_steps = num_accumulation_steps + 1
                # loss_this_step = loss_this_step + sess.run(loss_op, feed_dict = feed_dict)
                # loss_this_step = loss_this_step / num_accumulation_steps # average loss (over all slices of the image volume) in this step

            if args.accum_gradients == 1:    
                # ===========================
                # run accumulated_gradients_mean_op, with a value for the placeholder num_accumulation_steps_pl; followed by the train_op with applies the gradients
                # ===========================
                sess.run(accumulated_gradients_mean_op, feed_dict = {num_accumulation_steps_pl: num_accumulation_steps})
                # run the train_op.
                # this also requires input output placeholders, as compute_gradients will be called again..
                # But the returned gradient values will be replaced by the mean gradients.
                sess.run(train_op, feed_dict = feed_dict)

            # =============================
            # compute TTA loss after this epoch's parameter updates
            # =============================
            loss_iter_count = 0
            loss_this_step = 0.0
            for b_i in range(0, test_image.shape[0], b_size):
                if (b_i+1)*b_size > test_image.shape[0]:
                    continue
                loss_this_step = loss_this_step + sess.run(loss_op, feed_dict = {images_pl: np.expand_dims(test_image[b_i*b_size:(b_i+1)*b_size, :, :], axis=-1)})
                loss_iter_count += 1
            loss_this_step = loss_this_step / loss_iter_count
            summary_writer.add_summary(sess.run(loss_whole_subject_summary, feed_dict={loss_whole_subject_pl: loss_this_step}), step)

            # ===========================
            # compute an exponential moving average of the TTA loss
            # ===========================
            momentum = 0.95
            loss_ema = momentum * loss_ema + (1 - momentum) * loss_this_step
            summary_writer.add_summary(sess.run(loss_ema_summary, feed_dict={loss_ema_pl: loss_ema}), step)

            # ===========================
            # save models according to different criteria
            # ===========================
            if best_loss > loss_ema:
                best_loss = loss_ema
                best_file = os.path.join(log_dir_tta, 'models/best_loss.ckpt')
                saver_tta_best.save(sess, best_file, global_step=step)
                logging.info('Found new best score (%f) at step %d -  Saving model.' % (best_loss, step))

                # ===========================
                # save the best model in the first ten iterations.
                # ===========================
                if step < 11:
                    best_file = os.path.join(log_dir_tta, 'models/best_loss_in_first_10_epochs.ckpt')
                    saver_tta_best_first_10_epochs.save(sess, best_file, global_step=step)
                    logging.info('Found new best score (%f) at step %d -  Saving model.' % (best_loss, step))

                # ===========================
                # save the best model in the first 50 iterations.
                # ===========================
                if step < 51:
                    best_file = os.path.join(log_dir_tta, 'models/best_loss_in_first_50_epochs.ckpt')
                    saver_tta_best_first_50_epochs.save(sess, best_file, global_step=step)
                    logging.info('Found new best score (%f) at step %d -  Saving model.' % (best_loss, step))

                # ===========================
                # save the best model in the first 100 iterations.
                # ===========================
                if step < 101:
                    best_file = os.path.join(log_dir_tta, 'models/best_loss_in_first_100_epochs.ckpt')
                    saver_tta_best_first_100_epochs.save(sess, best_file, global_step=step)
                    logging.info('Found new best score (%f) at step %d -  Saving model.' % (best_loss, step))

            # ===========================
            # Save a 'sos' model if the average loss after the epoch is great than the average loss before the epoch
            # ===========================
            if loss_this_step > loss_previous_step and model_sos_saved == False:
                best_file = os.path.join(log_dir_tta, 'models/best_loss_sos.ckpt')
                saver_tta_sos.save(sess, best_file, global_step=step)
                logging.info('Loss increased in step %d as compared to the previous step -  Saving model.' % (step))
                model_sos_saved = True
            loss_previous_step = loss_this_step

            if step == 100 and model_sos_saved == False:
                best_file = os.path.join(log_dir_tta, 'models/best_loss_sos.ckpt')
                saver_tta_sos.save(sess, best_file, global_step=step)
                logging.info('Loss continuously decreased for the first %d epochs, saving current model as the SOS model.' % (step))
                model_sos_saved = True

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

            if b_size > 1 and test_image.shape[0] > b_size:
                label_predicted = np.reshape(label_predicted, (label_predicted.shape[0]*label_predicted.shape[1], label_predicted.shape[2], label_predicted.shape[3]))
                image_normalized = np.reshape(image_normalized, (image_normalized.shape[0]*image_normalized.shape[1], image_normalized.shape[2], image_normalized.shape[3]))
                
            label_predicted = label_predicted[:test_image.shape[0], ...]
            image_normalized = image_normalized[:test_image.shape[0], ...]

            if args.test_dataset in ['UCL', 'HK', 'BIDMC']:
                label_predicted[label_predicted!=0.0] = 1.0
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
                                                test_image_gt,
                                                test_image_gt)

            step = step + 1

        # ================================================================
        # close session
        # ================================================================
        sess.close()