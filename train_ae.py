# ==================================================================
# The following things happen in this file:
# ==================================================================
# 1. Get dataset dependent params (image_size, resolution, etc)
# 2. Load training data (for visualizing SD features as compared to TD features)
# 3. Set paths and directories for AE training
# 4. Build the TF graph (normalization, segmentation and AE networks)
# 5. Define loss functions for training the AE
# 6. Define optimizer
# 7. Define summary ops
# 8. Define savers
# 9. AE training iterations
# 10. Eval on training and validation sets
# 11. Visualize AE inputs and outputs
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

# Which features to autoencode
parser.add_argument('--ae_features', default = "xn") # xn | y | f1 | f2 | f3

# Batch settings
parser.add_argument('--b_size', type = int, default = 16)

# Learning rate settings
parser.add_argument('--ae_learning_rate', type = float, default = 0.0001) # 0.001 / 0.0005 / 0.0001 
parser.add_argument('--ae_runnum', type = int, default = 1) # 1 / 2 / 3

# parse arguments
args = parser.parse_args()

# ===========================
# Function for doing eval on a number of batches
# ===========================
def do_eval(sess,
            eval_loss,
            images_placeholder,
            training_time_placeholder,
            images,
            batch_size):
    
    loss_ii = 0
    num_batches_for_eval = 100
    for _ in range(num_batches_for_eval):
        random_indices = np.random.permutation(images.shape[0])
        batch_indices = np.sort(random_indices[:args.b_size]) # sort because hdf5 likes it like this
        x = np.expand_dims(images[batch_indices, ...], axis=-1)
        loss = sess.run(eval_loss, feed_dict={images_placeholder: x, training_time_placeholder: False})
        loss_ii += loss
    avg_loss = loss_ii / num_batches_for_eval
    logging.info('  Average recon loss: %.4f' % (avg_loss))
    return avg_loss

# ================================================================
# set dataset dependent hyperparameters
# ================================================================
dataset_params = exp_config.get_dataset_dependent_params(args.train_dataset) 
image_size = dataset_params[0]
nlabels = dataset_params[1]
target_resolution = dataset_params[2]
image_depth_tr = dataset_params[3]

# ================================================================
# load training data
# ================================================================
loaded_training_data = utils_data.load_training_data(args.train_dataset,
                                                     image_size,
                                                     target_resolution)
imtr = loaded_training_data[0]
imvl = loaded_training_data[9]

# dir where the SD mdoels have been saved
if args.train_dataset in ['UMC', 'site2']:
    expname_i2l = 'tr' + args.train_dataset + '_cv' + str(args.tr_cv_fold_num) + '_r' + str(args.tr_run_number) + '/' + 'i2i2l/'
else:
    expname_i2l = 'tr' + args.train_dataset + '_r' + str(args.tr_run_number) + '/' + 'i2i2l/'
log_dir = sys_config.project_root + 'log_dir/' + expname_i2l

# dir for AE
exp_str = 'tta/AE/r' + str(args.ae_runnum) + '/YufanArch/'
log_dir_ae = log_dir + exp_str
tensorboard_dir_ae = sys_config.tensorboard_root + expname_i2l + exp_str + args.ae_features + '/'

logging.info('SD training directory: %s' %log_dir)
logging.info('AE training directory: %s' %log_dir_ae)
logging.info('Tensorboard directory for AE training: %s' %tensorboard_dir_ae)

if not tf.gfile.Exists(log_dir_ae):
    tf.gfile.MakeDirs(log_dir_ae)
    tf.gfile.MakeDirs(log_dir_ae + 'models_' + args.ae_features + '/')
    tf.gfile.MakeDirs(tensorboard_dir_ae)

# ================================================================
# Build the TF graph
# ================================================================
with tf.Graph().as_default():
    
    # ======================
    # Set random seed for reproducibility
    # ======================
    tf.random.set_random_seed(args.ae_runnum)
    np.random.seed(args.ae_runnum)
    
    # ======================
    # create placeholders
    # ======================
    logging.info('Creating placeholders...')
    images_pl = tf.placeholder(tf.float32, shape=[exp_config.batch_size] + list(image_size) + [1], name = 'images')
    training_pl = tf.placeholder(tf.bool, shape=[], name = 'training_or_testing')
        
    # ======================
    # normalization module
    # ======================
    images_normalized, added_residual = model.normalize(images_pl,
                                                        exp_config,
                                                        training_pl = training_pl)
    
    # ======================
    # segmentation network
    # ======================
    logits, softmax, preds, features_level1, features_level2, features_level3 = model.predict_i2l(images_normalized,
                                                                                                  exp_config,
                                                                                                  training_pl = training_pl,
                                                                                                  nlabels = nlabels,
                                                                                                  return_features = True)

    logging.info('shape of input images: ' + str(images_pl.shape)) # (batch_size, 256, 256, 1)
    logging.info('shape of segmentation logits: ' + str(logits.shape)) # (batch_size, 256, 256, nlabels)
    logging.info('shape of features_level1: ' + str(features_level1.shape)) # (batch_size, 256, 256, nlabels)
    logging.info('shape of features_level2: ' + str(features_level2.shape)) # (batch_size, 256, 256, nlabels)
    logging.info('shape of features_level3: ' + str(features_level3.shape)) # (batch_size, 256, 256, nlabels)

    # ======================
    # define AEs
    # ======================
    if args.ae_features == 'xn':
        
        # ======================
        # autoencoder on the space of normalized images
        # ======================
        images_normalized_autoencoded = model.autoencode(images_normalized,
                                                         exp_config,
                                                         training_pl,
                                                         args.ae_features)
        logging.info('shape of normalized images: ' + str(images_normalized.shape)) # (batch_size, 256, 256, 1)
        logging.info('shape of autoencoded normalized images: ' + str(images_normalized_autoencoded.shape)) # (batch_size, 256, 256, 1)
        
        # ======================
        # AE loss
        # ======================
        loss_op = tf.reduce_mean(tf.math.square(images_normalized_autoencoded - images_normalized))
    
    elif args.ae_features == 'y':
        
        # ======================
        # autoencoder on the space of softmax output (combine all channels into one with the highest probability)
        # ======================
        softmax_autoencoded = model.autoencode(softmax,
                                               exp_config,
                                               training_pl,
                                               args.ae_features)
        logging.info('shape of AE input: ' + str(softmax.shape)) # (batch_size, 256, 256, num_classes)
        logging.info('shape of AE output: ' + str(softmax_autoencoded.shape)) # (batch_size, 256, 256, num_classes)

        # ======================
        # AE loss
        # ======================
        loss_op = tf.reduce_mean(tf.math.square(softmax_autoencoded - softmax))

    elif args.ae_features == 'f1':
        
        # ======================
        # autoencoder on the space of softmax output (combine all channels into one with the highest probability)
        # ======================
        features_level1_autoencoded = model.autoencode(features_level1,
                                                       exp_config,
                                                       training_pl,
                                                       args.ae_features)
        logging.info('shape of AE input: ' + str(features_level1.shape)) # (batch_size, 256, 256, num_classes)
        logging.info('shape of AE output: ' + str(features_level1_autoencoded.shape)) # (batch_size, 256, 256, num_classes)

        # ======================
        # AE loss
        # ======================
        loss_op = tf.reduce_mean(tf.math.square(features_level1_autoencoded - features_level1))

    elif args.ae_features == 'f2':
        
        # ======================
        # autoencoder on the space of softmax output (combine all channels into one with the highest probability)
        # ======================
        features_level2_autoencoded = model.autoencode(features_level2,
                                                       exp_config,
                                                       training_pl,
                                                       args.ae_features)
        logging.info('shape of AE input: ' + str(features_level2.shape)) # (batch_size, 256, 256, num_classes)
        logging.info('shape of AE output: ' + str(features_level2_autoencoded.shape)) # (batch_size, 256, 256, num_classes)

        # ======================
        # AE loss
        # ======================
        loss_op = tf.reduce_mean(tf.math.square(features_level2_autoencoded - features_level2))

    elif args.ae_features == 'f3':
        
        # ======================
        # autoencoder on the space of softmax output (combine all channels into one with the highest probability)
        # ======================
        features_level3_autoencoded = model.autoencode(features_level3,
                                                       exp_config,
                                                       training_pl,
                                                       args.ae_features)
        logging.info('shape of AE input: ' + str(features_level3.shape)) # (batch_size, 256, 256, num_classes)
        logging.info('shape of AE output: ' + str(features_level3_autoencoded.shape)) # (batch_size, 256, 256, num_classes)

        # ======================
        # AE loss
        # ======================
        loss_op = tf.reduce_mean(tf.math.square(features_level3_autoencoded - features_level3))
    
    # ======================
    # Add losses to tensorboard
    # ======================
    tf.summary.scalar('loss/AE', loss_op)         
    summary_during_ae_training = tf.summary.merge_all()

    # ======================
    # Divide the vars into normalization, segmentation and autoencoder networks
    # ======================
    all_vars = []
    seg_vars = []
    norm_vars = []
    i2l_vars = []
    ae_vars = []
    for v in tf.global_variables():
        var_name = v.name        
        all_vars.append(v)
        if 'image_normalizer' in var_name:
            norm_vars.append(v)
            i2l_vars.append(v)
        elif 'i2l_mapper' in var_name:
            seg_vars.append(v)
            i2l_vars.append(v)
        elif 'self_sup_ae' in var_name:
            ae_vars.append(v)

    debug_print_varnames = True
    if debug_print_varnames == True:
        logging.info('=========== Norm vars')
        for v in norm_vars: logging.info(v.name)
        logging.info('=========== Seg vars')
        for v in seg_vars: logging.info(v.name)
        logging.info('=========== AE vars')
        for v in ae_vars: logging.info(v.name)
        logging.info('=========== All vars')
        for v in all_vars: logging.info(v.name)
                
    # ======================
    # Add optimization ops.
    # ======================
    print('Creating AE training op...')
    train_op = model.training_step(loss_op,
                                   ae_vars,
                                   exp_config.optimizer_handle,
                                   args.ae_learning_rate,
                                   update_bn_nontrainable_vars=True)
       
    # ======================
    # Model evaluation
    # ======================
    print('creating eval op...')
    if args.ae_features == 'xn':
        eval_loss = model.evaluate_ae(images_normalized, images_normalized_autoencoded)
    elif args.ae_features == 'y':
        eval_loss = model.evaluate_ae(softmax, softmax_autoencoded)
    elif args.ae_features == 'f1':
        eval_loss = model.evaluate_ae(features_level1, features_level1_autoencoded)
    elif args.ae_features == 'f2':
        eval_loss = model.evaluate_ae(features_level2, features_level2_autoencoded)
    elif args.ae_features == 'f3':
        eval_loss = model.evaluate_ae(features_level3, features_level3_autoencoded)

    # ======================
    # build the summary Tensor based on the TF collection of Summaries.
    # ======================
    print('creating summary op...')
    summary = tf.summary.merge_all()

    # ======================
    # add init ops
    # ======================
    init_ops = tf.global_variables_initializer()
    
    # ======================
    # find if any vars are uninitialized
    # ======================
    logging.info('Adding the op to get a list of initialized variables...')
    uninit_vars = tf.report_uninitialized_variables()

    # ======================
    # create saver
    # ======================
    saver_i2l = tf.train.Saver(var_list = i2l_vars)
    saver = tf.train.Saver(var_list = ae_vars, max_to_keep=1)
    saver_best = tf.train.Saver(var_list = ae_vars, max_to_keep=1)
    
    # ======================
    # create session
    # ======================
    sess = tf.Session()

    # ======================
    # create a summary writer
    # ======================
    summary_writer = tf.summary.FileWriter(tensorboard_dir_ae, sess.graph)

    # ======================
    # summaries of the validation errors
    # ======================
    vl_error = tf.placeholder(tf.float32, shape=[], name='vl_error')
    vl_summary = tf.summary.scalar('validation/loss', vl_error)

    # ======================
    # summaries of the training errors
    # ======================        
    tr_error = tf.placeholder(tf.float32, shape=[], name='tr_error')
    tr_summary = tf.summary.scalar('training/loss', tr_error)
    
    # ======================
    # freeze the graph before execution
    # ======================
    logging.info('Freezing the graph now!')
    tf.get_default_graph().finalize()

    # ======================
    # Run the Op to initialize the variables.
    # ======================
    logging.info('============================================================')
    logging.info('initializing all variables...')
    sess.run(init_ops)
    
    # ======================
    # print names of uninitialized variables
    # ======================
    logging.info('============================================================')
    logging.info('This is the list of uninitialized variables:' )
    uninit_variables = sess.run(uninit_vars)
    for v in uninit_variables: print(v)

    # ======================
    # Restore trained normalization + segmentation network parameters
    # ======================    
    logging.info('============================================================')   
    path_to_model = sys_config.project_root + 'log_dir/' + expname_i2l + 'models/'
    checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_dice.ckpt')
    logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
    saver_i2l.restore(sess, checkpoint_path)

    # ======================
    # Initialize a step counter
    # ======================
    step = 0
    best_loss = 1000.0

    # ======================
    # run training epochs
    # ======================
    while (step < exp_config.max_steps_ae):

        if step % 1000 is 0:
            logging.info('============================================================')
            logging.info('step %d' % step)
    
        # ======================               
        # sample a random batch
        # ======================
        random_indices = np.random.permutation(imtr.shape[0])
        batch_indices = np.sort(random_indices[:args.b_size]) # sort because hdf5 likes it like this
        x = np.expand_dims(imtr[batch_indices, ...], axis=-1)
                    
        # ===========================
        # opt step
        # ===========================
        _, loss = sess.run([train_op, loss_op], feed_dict={images_pl: x, training_pl: True})

        # ===========================
        # write the summaries and print an overview fairly often
        # ===========================
        if (step+1) % exp_config.summary_writing_frequency == 0:                    
            logging.info('Step %d: loss = %.3f' % (step+1, loss))
            
            # ===========================
            # Update the events file
            # ===========================
            summary_str = sess.run(summary, feed_dict = {images_pl: x, training_pl: True})
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

        # ===========================
        # Compute the loss on the entire training set
        # ===========================
        if (step+1) % exp_config.train_eval_frequency == 0:
            logging.info('Training Data Eval:')
            train_loss = do_eval(sess,
                                 eval_loss,
                                 images_pl,
                                 training_pl,
                                 imtr,
                                 args.b_size)                    
            tr_summary_msg = sess.run(tr_summary, feed_dict={tr_error: train_loss})
            summary_writer.add_summary(tr_summary_msg, step)
                
        # ===========================
        # Save a checkpoint periodically
        # ===========================
        if step % exp_config.save_frequency == 0:
            checkpoint_file = os.path.join(log_dir_ae, 'models_' + args.ae_features + '/model.ckpt')
            saver.save(sess, checkpoint_file, global_step=step)

        # ===========================
        # Evaluate the model periodically on a validation set 
        # ===========================
        if (step+1) % exp_config.val_eval_frequency == 0:
            logging.info('Validation Data Eval:')
            val_loss = do_eval(sess,
                               eval_loss,
                               images_pl,
                               training_pl,
                               imvl,
                               args.b_size)                    
            vl_summary_msg = sess.run(vl_summary, feed_dict={vl_error: val_loss})
            summary_writer.add_summary(vl_summary_msg, step)

            # ===========================
            # save model if the val loss is the best yet
            # ===========================
            if val_loss < best_loss:
                best_loss = val_loss
                best_file = os.path.join(log_dir_ae, 'models_' + args.ae_features + '/best_loss.ckpt')
                saver_best.save(sess, best_file, global_step=step)
                logging.info('Found new average best loss on the validation set! - %f -  Saving model.' % val_loss)

        step += 1
                
    sess.close()