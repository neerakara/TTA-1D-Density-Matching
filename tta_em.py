# ==================================================================
# The following things happen in this file:
# ==================================================================
# 1. Get dataset dependent params (image_size, resolution, etc)
# 2. Load test data
# 3. Set paths and directories for the requested test ID
# 4. Extract test image for the requested test TD
# 5. Build the TF graph (normalization and segmentation networks)
# 6. Define loss function for entropy min.
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
parser.add_argument('--test_dataset', default = "USZ") # BMC / USZ / UCL / BIDMC / HK (prostate) | UHE / HVHD (cardiac) | UMC / NUHS (brain WMH) | CALTECH (brain tissues) | site3
parser.add_argument('--test_cv_fold_num', type = int, default = 1) # 1 / 2
parser.add_argument('--test_sub_num', type = int, default = 0) # 0 to 19

# TTA base string
parser.add_argument('--tta_string', default = "tta/")
# Which vars to adapt?
parser.add_argument('--TTA_VARS', default = "NORM") # BN / NORM

# Batch settings
parser.add_argument('--b_size', type = int, default = 8)

# Learning rate settings
parser.add_argument('--tta_learning_rate', type = float, default = 0.0001) # 0.001 / 0.0005 / 0.0001 
parser.add_argument('--tta_learning_sch', type = int, default = 0) # 0 / 1
parser.add_argument('--tta_runnum', type = int, default = 1) # 1 / 2 / 3

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
exp_str = exp_config.make_tta_exp_name(args, tta_method = 'entropy_min') + args.test_dataset + '_' + subject_name
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

        # ================================================================
        # Insert a normalization module in front of the segmentation network
        # the normalization module is adapted for each test image
        # ================================================================
        images_normalized, added_residual = model.normalize(images_pl, exp_config, training_pl = training_pl)
        
        # ================================================================
        # Build the graph that computes predictions from the inference model
        # ================================================================
        logits, softmax, preds = model.predict_i2l(images_normalized, exp_config, training_pl = training_pl, nlabels = nlabels)
        
        # ================================================================
        # Divide the vars into segmentation network and normalization network
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
        if args.TTA_VARS == "BN":
            tta_vars = bn_vars
        elif args.TTA_VARS == "NORM":
            tta_vars = normalization_vars

        # ================================================================
        # Self-Entropy Minimization per test volume
        # Shape of softmax is # batchsize * nx * ny * num_classes
        # ================================================================
        loss_op = tf.reduce_mean(tf.reduce_sum(- softmax * tf.math.log(softmax + 0.001), axis=-1))
                
        # ================================================================
        # add losses to tensorboard
        # ================================================================
        tf.summary.scalar('loss/TTA', loss_op)         
        summary_during_tta = tf.summary.merge_all()
        
        # ================================================================
        # Add optimization ops
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
        # placeholder for logging a smoothened loss
        # ================================================================                        
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
        saver_tta = tf.train.Saver(var_list = tta_vars, max_to_keep=1)   
        saver_tta_best = tf.train.Saver(var_list = tta_vars, max_to_keep=1)
                
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

        # ===================================
        # TTA / SFDA iterations
        # ===================================
        step = 0
        best_loss = 100000.0

        while (step < tta_max_steps):
            
            logging.info("TTA step: " + str(step+1))
            
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
            # Adaptation iterations within this epoch
            # =============================
            b_size = args.b_size
            for b_i in range(0, test_image.shape[0], b_size):

                feed_dict={images_pl: np.expand_dims(test_image[np.random.randint(0, test_image.shape[0], b_size), :, :], axis=-1),
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

            if b_size > 1 and test_image.shape[0] > b_size:
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