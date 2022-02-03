# ==================================================================
# The following things happen in this file:
# ==================================================================
# 1. Get dataset dependent params (image_size, resolution, etc)
# 2. Load test data
# 3. Set paths and directories for the requested test ID
# 4. Extract test image for the requested test TD
# 5. Build the TF graph (normalization, segmentation and denoising autoencoder networks)
# 6. Define loss function for DAE loss minimization
# 7. Define optimization routine (gradient aggregation over all batches to cover the image volume)
# 8. Define summary ops
# 9. Define savers
# 10. TTA iterations
    # A. Obtain current segmentation of the entire volume
    # B. Pass it through the DAE and obtain its denoised version
    # C. Pass the denoised segmentation via a placeholder and optimize the adaptable module using it.
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
parser.add_argument('--train_dataset', default = "HCPT1") # RUNMC (prostate) | CSF (cardiac) | UMC (brain white matter hyperintensities) | HCPT1 (brain subcortical tissues) | site2
parser.add_argument('--tr_run_number', type = int, default = 1) # 1 / 
parser.add_argument('--tr_cv_fold_num', type = int, default = 1) # 1 / 2
# Test dataset and subject number
parser.add_argument('--test_dataset', default = "CALTECH") # BMC / USZ / UCL / BIDMC / HK (prostate) | UHE / HVHD (cardiac) | UMC / NUHS (brain WMH) | CALTECH (brain tissues) | site3
parser.add_argument('--test_cv_fold_num', type = int, default = 1) # 1 / 2
parser.add_argument('--test_sub_num', type = int, default = 2) # 0 to 19

# TTA base string
parser.add_argument('--tta_string', default = "tta/")
parser.add_argument('--tta_method', default = "DAE")
parser.add_argument('--dae_runnum', type = int, default = 1) # 1 / 2 
# Which vars to adapt?
parser.add_argument('--TTA_VARS', default = "NORM") # BN / NORM / AdaptAx / AdaptAxAf

# Batch settings
parser.add_argument('--b_size', type = int, default = 8)
parser.add_argument('--b_size_dae', type = int, default = 1)

# Which vars to adapt?
parser.add_argument('--accum_gradients', type = int, default = 1) # 0 / 1

# Learning rate settings
parser.add_argument('--tta_learning_rate', type = float, default = 0.001) # 0.001 / 0.0005 / 0.0001 
parser.add_argument('--tta_learning_sch', type = int, default = 0) # 0 / 1
parser.add_argument('--tta_runnum', type = int, default = 1) # 1 / 2 / 3

# whether to print debug stuff or not
parser.add_argument('--debug', type = int, default = 0) # 1 / 0

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
if args.train_dataset == 'HCPT1': # for TTA-DAE, for brain datasets, we undersample in the z-direction, so need more iterations here.
    tta_max_steps = 4*dataset_params[6]
tta_model_saving_freq = dataset_params[7]
tta_vis_freq = dataset_params[8]
image_size_3d = dataset_params[11]
target_resolution_3d = dataset_params[12]

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

# ======================================
# Helper function (e.g. to predict segmentation / normalized image)
# ======================================
def predict(image, b_size, sess, images_pl, predict_op, dims):
    
    predicted = []
    
    for b_i in range(0, image.shape[0], b_size):
    
        if b_i + b_size < image.shape[0]:
            batch = np.expand_dims(image[b_i:b_i+b_size, ...], axis=-1)
    
        else:
            # pad zeros to have complete batches
            extra_zeros_needed = b_i + b_size - image.shape[0]
            batch = np.expand_dims(np.concatenate((image[b_i:, ...], np.zeros((extra_zeros_needed, image.shape[1], image.shape[2]))), axis=0), axis=-1)
    
        predicted.append(sess.run(predict_op, feed_dict={images_pl: batch}))
    
    predicted = np.squeeze(np.array(predicted)).astype(float)  

    if b_size > 1 and image.shape[0] > b_size:
        if dims == 5:
            predicted = np.reshape(predicted, (predicted.shape[0]*predicted.shape[1], predicted.shape[2], predicted.shape[3], predicted.shape[4]))
        elif dims == 4:
            predicted = np.reshape(predicted, (predicted.shape[0]*predicted.shape[1], predicted.shape[2], predicted.shape[3]))
    
    predicted = predicted[:image.shape[0], ...]

    return predicted
    
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

    logging.info('shape of the test image: ' + str(test_image.shape))
    logging.info('shape of the test GT: ' + str(test_image_gt.shape))

    # Beware: rescaling (especially downscaling) can sometimes introduce errors (especially in the segmentation space).
    test_image, test_image_gt = utils.rescale_image_and_label(test_image,
                                                              test_image_gt,
                                                              nlabels,
                                                              orig_data_res_z[sub_num],
                                                              new_resolution = target_resolution_3d[0],
                                                              new_depth = image_size_3d[0])

    logging.info('shape of the test image (after rescaling and cropping): ' + str(test_image.shape))
    logging.info('shape of the test GT (after rescaling and cropping): ' + str(test_image_gt.shape))
    # setting GT labels to zero to check if we get similar performance without access to the GT labels.
    # test_image_gt[test_image_gt!=0] = 0
    # Ran like this in tta_run_2. Got similar performance as run1. So the actual GT labels are not being used in any way to affect the TTA.
    # They are being used only for tracking the evolution of the actual Dice of interest.
    # TTA-DAE works really well for the prostate datasets!

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
        # Placeholder through which the DAE denoised labels are passed and used as pseudo ground truth labels for computing the TTA loss
        labels_dae_pl = tf.placeholder(tf.float32, shape = [args.b_size] + list(image_size) + [nlabels], name = 'labels_dae')        

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
            logits, softmax, preds = model.predict_i2l_with_adaptors(images_normalized,
                                                                     exp_config,
                                                                     training_pl = tf.constant(False, dtype=tf.bool),
                                                                     nlabels = nlabels)

        else: # Directly feed the input image to the normalization module
            # ================================================================
            # Insert a normalization module in front of the segmentation network
            # the normalization module is adapted for each test image
            # ================================================================ 
            images_normalized, added_residual = model.normalize(images_pl, exp_config, training_pl = tf.constant(False, dtype=tf.bool))

            # ================================================================
            # Build the graph that computes predictions from the inference model
            # ================================================================        
            logits, softmax, preds = model.predict_i2l(images_normalized,
                                                       exp_config,
                                                       training_pl = tf.constant(False, dtype=tf.bool),
                                                       nlabels = nlabels)

        # ================================================================
        # 3d DAE in the label space
        # ================================================================
        # predict the current segmentation for the entire volume, downsample it (for some anatomies) and pass it through this placeholder
        pred_seg_3d_pl = tf.placeholder(tf.uint8, shape = [args.b_size_dae] + list(image_size_3d), name = 'pred_seg_3d')
        pred_seg_3d_1hot_pl = tf.one_hot(pred_seg_3d_pl, depth = nlabels)
        
        # denoise the noisy segmentation
        _, pred_seg_3d_denoised_softmax, _ = model.predict_dae(pred_seg_3d_1hot_pl,
                                                               exp_config,
                                                               nlabels,
                                                               training_pl = tf.constant(False, dtype=tf.bool))
        
        # ================================================================
        # Divide the vars into different groups
        # ================================================================
        i2l_vars, normalization_vars, bn_vars, adapt_ax_vars, adapt_af_vars, dae_vars = model.divide_vars_into_groups(tf.global_variables(), DAE = True)

        if args.debug == 1:
            logging.info("Ax vars")
            for v in adapt_ax_vars:
                logging.info(v.name)
                logging.info(v.shape)
            
            logging.info("Af vars")
            for v in adapt_af_vars:
                logging.info(v.name)
                logging.info(v.shape)

            logging.info("Norm vars")
            for v in normalization_vars:
                logging.info(v.name)
                logging.info(v.shape)

            logging.info("DAE vars")
            for v in dae_vars:
                logging.info(v.name)
                logging.info(v.shape)

            logging.info("I2L vars")
            for v in i2l_vars:
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
        # The loss that will be minimized is the dice between the predictions and the dae outputs 
        # ================================================================        
        loss_dae_op = model.loss(logits = logits,
                                 labels = labels_dae_pl,
                                 nlabels = nlabels,
                                 loss_type = exp_config.loss_type_l2l,
                                 are_labels_1hot = True) 

        # ================================================================
        # TTA loss - spectral norm of the feature adaptors
        # ================================================================
        if args.TTA_VARS in ['AdaptAxAf', 'AdaptAx']:
            loss_spectral_norm_wf1_op = model.spectral_loss(wf1)
            loss_spectral_norm_wf2_op = model.spectral_loss(wf2)
            loss_spectral_norm_wf3_op = model.spectral_loss(wf3)
            loss_spectral_norm_op = loss_spectral_norm_wf1_op + loss_spectral_norm_wf2_op + loss_spectral_norm_wf3_op

            # ================================================================
            # Total TTA loss
            # ================================================================
            loss_op = loss_dae_op + args.lambda_spectral * loss_spectral_norm_op

        else:
            # ================================================================
            # Total TTA loss
            # ================================================================
            loss_op = loss_dae_op
                
        # ================================================================
        # add losses to tensorboard
        # ================================================================
        tf.summary.scalar('loss/TTA', loss_op)
        tf.summary.scalar('loss/TTA__DAE', loss_dae_op)
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
            
            # =========================
            # calculate gradients
            # tf.get_collection(tf.GraphKeys.UPDATE_OPS) does not return BN updates ops of S_theta, but does return those of H_psi!
            # We do not want to update any S_theta or H_psi parameters. 
            # We only want to update the BN params (if any) of N_phi.
            # (In any case, while the gradient wrt N_phi are accumulated, the graph is not connected with H_psi. So it shouldn't matter even if we do run the update ops.)
            # (Perhaps, this is what ensures no update of H_psi BN params in the original implementation).
            # =========================
            # update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            # logging.info("----------------------- UPDATE OPS")
            # for op in update_ops: logging.info(op.name)
            # with tf.control_dependencies(update_ops):
                # gradients = optimizer.compute_gradients(loss_op, var_list = tta_vars)

            # =========================
            # calculate gradients
            # =========================
            gradients = optimizer.compute_gradients(loss_op, var_list = tta_vars)
            
            # =========================
            # define accumulation ops
            # =========================
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
        gt_dice_pl = tf.placeholder(tf.float32, shape=[], name='gt_dice')
        gt_dice_summary = tf.summary.scalar('TTA/gt_dice', gt_dice_pl)
        dae_dice_pl = tf.placeholder(tf.float32, shape=[], name='dae_dice')
        dae_dice_summary = tf.summary.scalar('TTA/dae_dice', dae_dice_pl)

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
        saver_dae = tf.train.Saver(var_list = dae_vars)
        # tta savers (we need multiple of these to save according to different stopping criteria)
        saver_tta = tf.train.Saver(var_list = tta_vars, max_to_keep=1) # general saver after every few epochs
        saver_tta_init = tf.train.Saver(var_list = tta_vars, max_to_keep=1) # saves the initial values of the TTA vars
        saver_tta_best = tf.train.Saver(var_list = tta_vars, max_to_keep=1) # saves the weights when the exponential moving average of the loss is the lowest
                
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
        # Restore the denoising autoencoder parameters
        # ================================================================
        path_to_dae_model = sys_config.project_root + 'log_dir/' + expname_i2l + 'tta/DAE/r' + str(args.dae_runnum) + '/models/'
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_dae_model, 'best_loss.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_dae.restore(sess, checkpoint_path)

        # ===================================
        # TTA / SFDA iterations
        # ===================================
        b_size = args.b_size
        best_dae_dice = 0.0
        step = 1

        while (step < tta_max_steps):
            
            # ======================================================
            # Book-keeping
            # ======================================================
            # 1. Compute the segmentation for the whole volume at this step
            # 2. Pass this through the DAE and get the denoised segmentation at this step
            # 3. Compute dice
            #   A. between the current prediction and the GT
            #   B. between the current prediction and the DAE denoised version of it.
            # 4. If 3B is the best yet, save the model at this state 
            # 5. Visualize the image, normalized image and the predicted segmentation
            # ======================================================
            if step == 1 or step % tta_vis_freq == 0:

                label_predicted_soft = predict(test_image, b_size, sess, images_pl, softmax, dims=5)
                label_predicted_hard = np.argmax(label_predicted_soft, axis=-1)

                # ==================
                # denoise the predicted 3D segmentation using the DAE
                # ==================
                feed_dict = {pred_seg_3d_1hot_pl: np.expand_dims(label_predicted_soft, axis=0)}                 
                y_pred_noisy_denoised_soft = np.squeeze(sess.run(pred_seg_3d_denoised_softmax, feed_dict=feed_dict)).astype(np.float16)               
                y_pred_noisy_denoised_hard = np.argmax(y_pred_noisy_denoised_soft, axis=-1)
                dae_dice = np.mean(met.f1_score(label_predicted_hard.flatten(), y_pred_noisy_denoised_hard.flatten(), average=None)[1:])

                if args.test_dataset in ['UCL', 'HK', 'BIDMC']:
                    label_predicted_hard[label_predicted_hard!=0.0] = 1.0
                dice_wrt_gt = np.mean(met.f1_score(test_image_gt.flatten(), label_predicted_hard.flatten(), average=None)[1:]) 
                
                # ==================
                # Record dice before this step's updates occur
                # ==================
                summary_writer.add_summary(sess.run(gt_dice_summary, feed_dict={gt_dice_pl: dice_wrt_gt}), step-1)
                summary_writer.add_summary(sess.run(dae_dice_summary, feed_dict={dae_dice_pl: dae_dice}), step-1)
                
                # log
                logging.info('Dice (prediction, ground truth) at TTA step ' + str(step-1) + ': ' + str(np.round(dice_wrt_gt, 3)))
                logging.info('Dice (prediction, DAE output) at TTA step ' + str(step-1) + ': ' + str(np.round(dae_dice, 3)))

                if step == 1:
                    # save initial values of the TTA vars (useful to check how much does the performance degrade due to random init of the adaptor modules)
                    saver_tta_init.save(sess, os.path.join(log_dir_tta, 'models/tta_init.ckpt'), global_step=0)

                # ==================
                # save best model so far
                # ==================
                if dae_dice > best_dae_dice:
                    best_dae_dice = dae_dice
                    best_file = os.path.join(log_dir_tta, 'models/best_dice.ckpt')
                    saver_tta_best.save(sess, best_file, global_step=step-1)
                    logging.info('Found new best score (%f) at step %d -  Saving model.' % (best_dae_dice, step-1))

                # ===========================   
                # visualize 
                # ===========================
                image_normalized = predict(test_image, b_size, sess, images_pl, images_normalized, dims=4)
                utils_vis.write_image_summaries(step-1,
                                                summary_writer,
                                                sess,
                                                images_summary,
                                                display_pl,
                                                test_image,
                                                image_normalized,
                                                label_predicted_hard,
                                                test_image_gt,
                                                test_image_gt)                    
            
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

                batch_indices = np.random.randint(0, test_image.shape[0], b_size)
                feed_dict = {images_pl: np.expand_dims(test_image[batch_indices, :, :], axis=-1),
                             labels_dae_pl: y_pred_noisy_denoised_soft[batch_indices, ...],
                             lr_pl: tta_learning_rate}
                
                # run the accumulate gradients op 
                if args.accum_gradients == 1:    
                    sess.run(accumulate_gradients_op, feed_dict=feed_dict)
                # run the train op on this batch
                elif args.accum_gradients == 0:    
                    sess.run(train_op, feed_dict=feed_dict)
                                    
                # increment the counter for the number of accumulations done so far in this 'epoch'
                num_accumulation_steps = num_accumulation_steps + 1

            if args.accum_gradients == 1:    
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
            if step % tta_model_saving_freq == 0:
                saver_tta.save(sess, os.path.join(log_dir_tta, 'models/model.ckpt'), global_step=step)

            # ===========================
            # Update the events file
            # ===========================
            summary_str = sess.run(summary_during_tta, feed_dict = feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            step = step + 1

        # ================================================================
        # close session
        # ================================================================
        sess.close()