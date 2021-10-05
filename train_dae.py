# ==================================================================
# The following things happen in this file:
# ==================================================================
# 1. Get dataset dependent params (image_size, resolution, etc)
# 2. Load training data
# 3. Set paths and directories for DAE training
# 4. Build the TF graph (the DAE network)
# 5. Define loss functions for training the DAE
# 6. Define optimizer
# 7. Define summary ops
# 8. Define savers
# 9. DAE training iterations
# 10. Eval on training and validation sets
# 11. Visualize DAE inputs and outputs
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

import utils
import utils_vis
import utils_data
import model as model
import config.params as exp_config
import config.system_paths as sys_config

# add to the top of your code under import tensorflow as tf
# config = tf.ConfigProto()
# config.gpu_options.allow_growth = True

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

# Batch settings
parser.add_argument('--b_size', type = int, default = 1)

# mask settings
parser.add_argument('--mask_type', default = "squares_jigsaw")
parser.add_argument('--mask_radius', type = int, default = 10)
parser.add_argument('--num_squares', type = int, default = 200)

# data aug settings
parser.add_argument('--da_ratio', type = float, default = 0.25)
parser.add_argument('--sigma', type = int, default = 20)
parser.add_argument('--alpha', type = int, default = 1000)
parser.add_argument('--trans_min', type = int, default = -10)
parser.add_argument('--trans_max', type = int, default = 10)
parser.add_argument('--rot_min', type = int, default = -10)
parser.add_argument('--rot_max', type = int, default = 10)
parser.add_argument('--scale_min', type = float, default = 0.9)
parser.add_argument('--scale_max', type = float, default = 1.1)

# Learning rate settings
parser.add_argument('--dae_learning_rate', type = float, default = 0.0001) # 0.001 / 0.0005 / 0.0001 
parser.add_argument('--dae_runnum', type = int, default = 1) # 1 / 2 / 3

# parse arguments
args = parser.parse_args()

# ================================================================
# dir where the SD mdoels have been saved
# ================================================================
expname_i2l = 'tr' + args.train_dataset + '_cv' + str(args.tr_cv_fold_num) + '_r' + str(args.tr_run_number) + '/' + 'i2i2l/'
log_dir = sys_config.project_root + 'log_dir/' + expname_i2l

# ================================================================
# dir for DAE
# ================================================================
exp_str = 'tta/DAE/r' + str(args.dae_runnum) + '/'
log_dir_dae = log_dir + exp_str
tensorboard_dir_dae = sys_config.tensorboard_root + expname_i2l + exp_str + '/'
logging.info('SD training directory: %s' %log_dir)
logging.info('DAE training directory: %s' %log_dir_dae)
logging.info('Tensorboard directory for DAE training: %s' %tensorboard_dir_dae)
if not tf.gfile.Exists(log_dir_dae):
    tf.gfile.MakeDirs(log_dir_dae)
    tf.gfile.MakeDirs(log_dir_dae + 'models/')
    tf.gfile.MakeDirs(tensorboard_dir_dae)

# ================================================================
# set dataset dependent hyperparameters
# ================================================================
dataset_params = exp_config.get_dataset_dependent_params(args.train_dataset) 
nlabels = dataset_params[1]
image_size_3d = dataset_params[11]
target_resolution_3d = dataset_params[12]

# ================================================================
# load training data
# ================================================================
loaded_training_data = utils_data.load_labels_3d(args.train_dataset, image_size_3d, target_resolution_3d)
gttr = loaded_training_data[0]
gtvl = loaded_training_data[1]

logging.info('Training Labels: %s' %str(gttr.shape)) # expected: [num_subjects, img_size_z, img_size_x, img_size_y]
logging.info('Validation Labels: %s' %str(gtvl.shape))

# ==========================
# visualize
# ==========================
visualize_training_data = False
if visualize_training_data:
    for subject_num in range(gttr.shape[0]):
        utils_vis.save_samples_downsampled(gttr[subject_num, ...], savepath = log_dir_dae + 'tr_image_' + str(subject_num+1) + '.png')

# ===========================
# Function for doing eval on a number of batches
# ===========================
def do_eval(sess,
            eval_loss,
            true_labels_placeholder,
            blank_masks_placeholder,
            wrong_labels_placeholder,
            training_time_placeholder,
            labels,
            batch_size):

    loss = 0
    num_batches_for_eval = 100
    for _ in range(num_batches_for_eval):
        true_labels_eval, blank_masks_eval, wrong_labels_eval = get_batch(labels, batch_size)
        feed_dict = {true_labels_placeholder: true_labels_eval,
                     blank_masks_placeholder: blank_masks_eval,
                     wrong_labels_placeholder: wrong_labels_eval,
                     training_time_placeholder: False}
        loss += sess.run(eval_loss, feed_dict=feed_dict)
    avg_loss = loss / num_batches_for_eval
    logging.info('  Average segmentation loss: %.4f' % (avg_loss))
    return avg_loss

# ==================================================================
# ==================================================================
def get_batch(labels,
              batch_size,
              train_or_eval = 'train'):

    # ===========================
    # generate indices to randomly select subjects in each minibatch
    # ===========================
    nun_subjects = labels.shape[0]
    random_indices = np.random.permutation(nun_subjects)
    batch_indices = np.sort(random_indices[:batch_size])
    labels_this_batch = labels[batch_indices, ...]
    
    # ===========================    
    # data augmentation (random elastic transformations, translations, rotations, scaling)
    # doing data aug both during training as well as during evaluation on the validation set (used for model selection)
    # ===========================                  
    labels_this_batch = utils.do_data_augmentation_on_3d_labels(labels = labels_this_batch,
                                                                data_aug_ratio = args.da_ratio,
                                                                sigma = args.sigma,
                                                                alpha = args.alpha,
                                                                trans_min = args.trans_min,
                                                                trans_max = args.trans_max,
                                                                rot_min = args.rot_min,
                                                                rot_max = args.rot_max,
                                                                scale_min = args.scale_min,
                                                                scale_max = args.scale_max)
    
    # ==================    
    # make labels 1-hot
    # ==================
    labels_this_batch_1hot = utils.make_onehot(labels_this_batch, nlabels)
                
    # ===========================      
    # make noise masks that the autoencoder with try to denoise
    # ===========================      
    if train_or_eval is 'train':
        blank_masks_this_batch, wrong_labels_this_batch = utils.make_noise_masks_3d(shape = [args.b_size] + list(image_size_3d) + [nlabels],
                                                                                    mask_type = args.mask_type,
                                                                                    mask_params = [args.mask_radius, args.num_squares],
                                                                                    nlabels = nlabels,
                                                                                    labels_1hot = labels_this_batch_1hot)
        
    elif train_or_eval is 'eval':
        # fixing amount of noise in order to get comparable runs during evaluation
        blank_masks_this_batch, wrong_labels_this_batch = utils.make_noise_masks_3d(shape = [args.b_size] + list(image_size_3d) + [nlabels],
                                                                                    mask_type = args.mask_type,
                                                                                    mask_params = [args.mask_radius, args.num_squares],
                                                                                    nlabels = nlabels,
                                                                                    labels_1hot = labels_this_batch_1hot,
                                                                                    is_num_masks_fixed = True, # fixing amount of noise in order to get comparable runs during evaluation
                                                                                    is_size_masks_fixed = True) # fixing amount of noise in order to get comparable runs during evaluation

    return labels_this_batch, blank_masks_this_batch, wrong_labels_this_batch

# ================================================================
# Build the TF graph
# ================================================================
with tf.Graph().as_default():
    
    # ======================
    # Set random seed for reproducibility
    # ======================
    tf.random.set_random_seed(args.dae_runnum)
    np.random.seed(args.dae_runnum)
            
    # ================================================================
    # create placeholders
    # ================================================================
    logging.info('Creating placeholders...')        
    true_labels_shape = [args.b_size] + list(image_size_3d)
    true_labels_pl = tf.placeholder(tf.uint8, shape = true_labels_shape, name = 'true_labels')
    
    # ================================================================
    # This will be a mask with all zeros in locations of pixels that we want to alter the labels of.
    # Multiply with this mask to have zero vectors for all those pixels.
    # ================================================================        
    blank_masks_shape = [args.b_size] + list(image_size_3d) + [nlabels]
    blank_masks_pl = tf.placeholder(tf.float32, shape = blank_masks_shape, name = 'blank_masks')
    
    # ================================================================
    # This will be a mask with all zeros in locations of pixels that we want to alter the labels of.
    # Multiply with this mask to have zero vectors for all those pixels.
    # ================================================================        
    wrong_labels_shape = [args.b_size] + list(image_size_3d) + [nlabels]
    wrong_labels_pl = tf.placeholder(tf.float32, shape = wrong_labels_shape, name = 'wrong_labels')

    # ================================================================        
    # Training placeholder
    # ================================================================        
    training_pl = tf.placeholder(tf.bool, shape=[], name = 'training_or_testing')

    # ================================================================
    # make true labels 1-hot
    # ================================================================
    true_labels_1hot = tf.one_hot(true_labels_pl, depth = nlabels)
    
    # ================================================================
    # Blank certain locations and write wrong labels in those locations
    # ================================================================
    noisy_labels_1hot = tf.math.multiply(true_labels_1hot, blank_masks_pl) + wrong_labels_pl
            
    # ================================================================
    # build the graph that computes predictions from the inference model
    # ================================================================
    autoencoded_logits, _, _ = model.predict_dae(noisy_labels_1hot,
                                                 exp_config,
                                                 nlabels,
                                                 training_pl = training_pl)

    print('shape of input tensor: ', true_labels_pl.shape) # (batch_size, nz, nx, ny)
    print('shape of input tensor converted to 1-hot: ', true_labels_1hot.shape) # (batch_size, nz, nx, ny, nlabels) 
    print('shape of predicted logits: ', autoencoded_logits.shape) # (batch_size, nz, nx, ny, nlabels) 

    # ================================================================
    # create a list of all vars that must be optimized wrt
    # ================================================================
    l2l_vars = []
    for v in tf.trainable_variables():
        print(v.name)
        l2l_vars.append(v)
    
    # ================================================================
    # add ops for calculation of the supervised training loss
    # ================================================================
    loss_op = model.loss(logits = autoencoded_logits,
                         labels = true_labels_1hot,
                         nlabels = nlabels,
                         loss_type = exp_config.loss_type_l2l,
                         are_labels_1hot = True)
    tf.summary.scalar('loss', loss_op)
    
    # ================================================================
    # add optimization ops.
    # Create different ops according to the variables that must be trained
    # ================================================================
    print('creating training op...')
    train_op = model.training_step(loss_op,
                                   l2l_vars,
                                   exp_config.optimizer_handle,
                                   args.dae_learning_rate,
                                   update_bn_nontrainable_vars=True)

    # ================================================================
    # add ops for model evaluation
    # ================================================================
    print('creating eval op...')
    eval_loss = model.evaluation_dae(clean_labels = true_labels_1hot,
                                     noisy_labels = noisy_labels_1hot,
                                     denoised_logits = autoencoded_logits,
                                     nlabels = nlabels,
                                     loss_type = exp_config.loss_type_l2l,
                                     are_labels_1hot = True)

    # ================================================================
    # build the summary Tensor based on the TF collection of Summaries.
    # ================================================================
    print('creating summary op...')
    summary = tf.summary.merge_all()

    # ================================================================
    # add init ops
    # ================================================================
    init_ops = tf.global_variables_initializer()
    
    # ================================================================
    # find if any vars are uninitialized
    # ================================================================
    logging.info('Adding the op to get a list of initialized variables...')
    uninit_vars = tf.report_uninitialized_variables()

    # ================================================================
    # create saver
    # ================================================================
    saver = tf.train.Saver(max_to_keep=1)
    saver_best_dice = tf.train.Saver(max_to_keep=1)
    
    # ================================================================
    # create session
    # ================================================================
    sess = tf.Session()
    # sess = tf.Session(config=config)

    # ================================================================
    # create a summary writer
    # ================================================================
    summary_writer = tf.summary.FileWriter(tensorboard_dir_dae, sess.graph)

    # ================================================================
    # summaries of the validation errors
    # ================================================================
    vl_loss = tf.placeholder(tf.float32, shape=[], name='vl_loss')
    vl_loss_summary = tf.summary.scalar('validation/loss', vl_loss)

    # ================================================================
    # summaries of the training errors
    # ================================================================        
    tr_loss = tf.placeholder(tf.float32, shape=[], name='tr_loss')
    tr_loss_summary = tf.summary.scalar('training/loss', tr_loss)
    
    # ================================================================
    # freeze the graph before execution
    # ================================================================
    logging.info('Freezing the graph now!')
    tf.get_default_graph().finalize()

    # ================================================================
    # Run the Op to initialize the variables.
    # ================================================================
    logging.info('============================================================')
    logging.info('initializing all variables...')
    sess.run(init_ops)

    # ================================================================
    # print names of all variables
    # ================================================================
    logging.info('============================================================')
    logging.info('This is the list of all variables:' )
    for v in tf.trainable_variables(): print(v.name)
    
    # ================================================================
    # print names of uninitialized variables
    # ================================================================
    logging.info('============================================================')
    logging.info('This is the list of uninitialized variables:' )
    uninit_variables = sess.run(uninit_vars)
    for v in uninit_variables: print(v)

    # ================================================================
    # ================================================================        
    step = 0
    best_loss = 10.0

    # ================================================================
    # run training epochs
    # ================================================================
    while (step < exp_config.max_steps_dae):

        if step % 1000 is 0:
            logging.info('============================================================')
            logging.info('step %d' % step)
    
        # ================================================               
        # batches
        # ================================================
        true_labels, blank_masks, wrong_labels = get_batch(gttr, args.b_size)            
            
        # ===========================
        # create the feed dict for this training iteration
        # ===========================
        feed_dict = {true_labels_pl: true_labels,
                     blank_masks_pl: blank_masks,
                     wrong_labels_pl: wrong_labels, 
                     training_pl: True}
        
        # ===========================
        # opt step
        # ===========================
        _, loss = sess.run([train_op, loss_op], feed_dict=feed_dict)

        # ===========================
        # write the summaries and print an overview fairly often
        # ===========================
        if (step+1) % exp_config.summary_writing_frequency == 0:                    
            logging.info('Step %d: loss = %.3f' % (step+1, loss))
            
            # ===========================
            # Update the events file
            # ===========================
            summary_str = sess.run(summary, feed_dict = feed_dict)
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        # ===========================
        # Compute the loss on the entire training set
        # ===========================
        if step % exp_config.train_eval_frequency == 0:
            logging.info('Training Data Eval:')
            train_loss = do_eval(sess,
                                 eval_loss,
                                 true_labels_pl,
                                 blank_masks_pl,
                                 wrong_labels_pl,
                                 training_pl,
                                 gttr,
                                 args.b_size)                    
            tr_summary_msg = sess.run(tr_loss_summary, feed_dict={tr_loss: train_loss})
            summary_writer.add_summary(tr_summary_msg, step)
            
        # ===========================
        # Save a checkpoint periodically
        # ===========================
        if step % exp_config.save_frequency == 0:
            checkpoint_file = os.path.join(log_dir_dae, 'models/model.ckpt')
            saver.save(sess, checkpoint_file, global_step=step)

        # ===========================
        # Evaluate the model periodically on a validation set 
        # ===========================
        if step % exp_config.val_eval_frequency == 0:
            logging.info('Validation Data Eval:')
            val_loss = do_eval(sess,
                               eval_loss,
                               true_labels_pl,
                               blank_masks_pl,
                               wrong_labels_pl,
                               training_pl,
                               gtvl,
                               args.b_size)
            vl_summary_msg = sess.run(vl_loss_summary, feed_dict={vl_loss: val_loss})
            summary_writer.add_summary(vl_summary_msg, step)

            # ===========================
            # save model if the val dice is the best yet
            # ===========================
            if val_loss < best_loss:
                best_loss = val_loss
                best_file = os.path.join(log_dir_dae, 'models/best_loss.ckpt')
                saver_best_dice.save(sess, best_file, global_step=step)
                logging.info('Found new average best loss on validation sets! - %f -  Saving model.' % val_loss)

        step += 1
            
    sess.close()