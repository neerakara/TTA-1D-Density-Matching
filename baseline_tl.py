# ==================================================================
# TRANSFER LEARNING BENCHMARK
# ==================================================================

# ==================================================================
# import 
# ==================================================================
import logging
import os.path
import time
import shutil
import tensorflow as tf
import numpy as np
import utils
import utils_data
import model as model
import config.params as exp_config
import config.system_paths as sys_config
import argparse

# ==================================================================
# parse arguments
# ==================================================================
parser = argparse.ArgumentParser(prog = 'PROG')

# Training dataset and run number
parser.add_argument('--train_dataset', default = "HCPT1") # RUNMC (prostate) | CSF (cardiac) | UMC (brain white matter hyperintensities) | HCPT1 (brain subcortical tissues)
parser.add_argument('--tr_run_number', type = int, default = 1) # 1 / 
parser.add_argument('--tr_cv_fold_num', type = int, default = 1) # 1 / 2
# Test dataset and subject number
parser.add_argument('--test_dataset', default = "CALTECH") # BMC / USZ / UCL / BIDMC / HK (prostate) | UHE / HVHD (cardiac) | UMC / NUHS (brain WMH) | CALTECH (brain tissues)
parser.add_argument('--test_cv_fold_num', type = int, default = 1) # 1 / 2

# Batch settings
parser.add_argument('--b_size', type = int, default = 16)


# TL base string
parser.add_argument('--TL_STRING', default = "tl/")
# Which vars to adapt?
parser.add_argument('--TL_VARS', default = "ALL") # BN / NORM / ALL
# TL run number
parser.add_argument('--tl_runnum', type = int, default = 1) # 1 / 2 / 3

# parse arguments
args = parser.parse_args()

# ================================================================
# set dataset dependent hyperparameters
# ================================================================
logging.info('TRANSFER LEARNING')
logging.info('SD: ' + str(args.train_dataset))
logging.info('TD: ' + str(args.test_dataset))
dataset_params = exp_config.get_dataset_dependent_params(args.train_dataset, args.test_dataset) 
image_size = dataset_params[0]
nlabels = dataset_params[1]
target_resolution = dataset_params[2]

# ================================================================
# Set paths and directories
# ================================================================
# dir where the SD mdoels have been saved
expname_i2l = 'tr' + args.train_dataset + '_cv' + str(args.tr_cv_fold_num) + '_r' + str(args.tr_run_number) + '/' + 'i2i2l/'
log_dir_sd = sys_config.project_root + 'log_dir/' + expname_i2l

# dir for TL
exp_str = exp_config.make_tl_exp_name(args)
log_dir_tl = log_dir_sd + exp_str
tensorboard_dir_tl = sys_config.tensorboard_root + expname_i2l + exp_str

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
logging.info('SD training directory: %s' %log_dir_sd)
logging.info('Transfer learning directory: %s' %log_dir_tl)
logging.info('Tensorboard directory Transfer Learning: %s' %tensorboard_dir_tl)

# ==================================================================
# main function for transfer learning
# ==================================================================
def run_transfer():

    # ============================   
    # Load training data of the test distribution
    # ============================   
    loaded_training_data = utils_data.load_training_data(args.test_dataset,
                                                         image_size,
                                                         target_resolution,
                                                         args.test_cv_fold_num)

    imtr = loaded_training_data[0]
    gttr = loaded_training_data[1]
    imvl = loaded_training_data[9]
    gtvl = loaded_training_data[10]
                
    logging.info('Training Images: %s' %str(imtr.shape)) # expected: [num_slices, img_size_x, img_size_y]
    logging.info('Training Labels: %s' %str(gttr.shape)) # expected: [num_slices, img_size_x, img_size_y]
    logging.info('Validation Images: %s' %str(imvl.shape))
    logging.info('Validation Labels: %s' %str(gtvl.shape))
    logging.info('============================================================')
                
    # ================================================================
    # build the TF graph
    # ================================================================
    with tf.Graph().as_default():
        
        # ============================
        # set random seed for reproducibility
        # ============================
        tf.random.set_random_seed(args.tl_runnum)
        np.random.seed(args.tl_runnum)

        # ================================================================
        # create placeholders
        # ================================================================
        logging.info('Creating placeholders...')
        image_tensor_shape = [exp_config.batch_size] + list(image_size) + [1]
        mask_tensor_shape = [exp_config.batch_size] + list(image_size)
        images_pl = tf.placeholder(tf.float32, shape=image_tensor_shape, name = 'images')
        labels_pl = tf.placeholder(tf.uint8, shape=mask_tensor_shape, name = 'labels')
        learning_rate_pl = tf.placeholder(tf.float32, shape=[], name = 'learning_rate')
        training_pl = tf.placeholder(tf.bool, shape=[], name = 'training_or_testing')
        
        # ================================================================
        # insert a normalization module in front of the segmentation network
        # the normalization module will be adapted for each test image
        # ================================================================
        images_normalized, _ = model.normalize(images_pl, exp_config, training_pl)

        # ================================================================
        # build the graph that computes predictions from the inference model
        # ================================================================
        logits, _, _ = model.predict_i2l(images_normalized, exp_config, training_pl = training_pl, nlabels = nlabels)
        
        print('shape of inputs: ', images_pl.shape) # (batch_size, 256, 256, 1)
        print('shape of logits: ', logits.shape) # (batch_size, 256, 256, nlabels)
        
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
        if args.TL_VARS == "ALL":
            tl_vars = i2l_vars
        elif args.TL_VARS == "BN":
            tl_vars = bn_vars
        elif args.TL_VARS == "NORM":
            tl_vars = normalization_vars
        
        # ================================================================
        # add ops for calculation of the supervised training loss
        # ================================================================
        loss_op = model.loss(logits, labels_pl, nlabels=nlabels, loss_type=exp_config.loss_type)        
        tf.summary.scalar('loss', loss_op)
        
        # ================================================================
        # add optimization ops.
        # Create different ops according to the variables that must be trained
        # ================================================================
        print('creating training op...')
        train_op = model.training_step(loss_op, tl_vars, exp_config.optimizer_handle, learning_rate_pl,update_bn_nontrainable_vars=True)

        # ================================================================
        # add ops for model evaluation
        # ================================================================
        print('creating eval op...')
        eval_loss = model.evaluation_i2l(logits, labels_pl, images_pl, nlabels = nlabels, loss_type = exp_config.loss_type)

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
        saver = tf.train.Saver(var_list = i2l_vars, max_to_keep=1)
        saver_tl = tf.train.Saver(var_list = tl_vars, max_to_keep=1)
        saver_tl_best = tf.train.Saver(var_list = tl_vars, max_to_keep=1)
        
        # ================================================================
        # create session
        # ================================================================
        sess = tf.Session()

        # ================================================================
        # create a summary writer
        # ================================================================
        summary_writer = tf.summary.FileWriter(tensorboard_dir_tl, sess.graph)

        # ================================================================
        # summaries of the validation errors
        # ================================================================
        vl_error = tf.placeholder(tf.float32, shape=[], name='vl_error')
        vl_error_summary = tf.summary.scalar('validation/loss', vl_error)
        vl_dice = tf.placeholder(tf.float32, shape=[], name='vl_dice')
        vl_dice_summary = tf.summary.scalar('validation/dice', vl_dice)
        vl_summary = tf.summary.merge([vl_error_summary, vl_dice_summary])

        # ================================================================
        # summaries of the training errors
        # ================================================================        
        tr_error = tf.placeholder(tf.float32, shape=[], name='tr_error')
        tr_error_summary = tf.summary.scalar('training/loss', tr_error)
        tr_dice = tf.placeholder(tf.float32, shape=[], name='tr_dice')
        tr_dice_summary = tf.summary.scalar('training/dice', tr_dice)
        tr_summary = tf.summary.merge([tr_error_summary, tr_dice_summary])
        
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
        # load the SD trained weights
        # ================================================================
        logging.info('============================================================')   
        path_to_model = log_dir_sd + 'models/'
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_dice.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver.restore(sess, checkpoint_path)

        # ================================================================
        # Initiate counters
        # ================================================================        
        step = 0
        best_dice = 0

        # ================================================================
        # run training epochs
        # ================================================================
        while (step < exp_config.max_steps_tl):

            if step % 1000 is 0:
                logging.info('============================================================')
                logging.info('step %d' % step)
        
            # ================================================               
            # batches
            # ================================================            
            for batch in iterate_minibatches(imtr, gttr, batch_size = exp_config.batch_size, train_or_eval = 'train'):
                
                start_time = time.time()
                x, y = batch

                # ===========================
                # avoid incomplete batches
                # ===========================
                if y.shape[0] < exp_config.batch_size:
                    step += 1
                    continue
                
                # ===========================
                # create the feed dict for this training iteration
                # ===========================
                feed_dict = {images_pl: x, labels_pl: y, learning_rate_pl: exp_config.learning_rate_tl, training_pl: True}
                
                # ===========================
                # opt step
                # ===========================
                _, loss = sess.run([train_op, loss_op], feed_dict=feed_dict)

                # ===========================
                # compute the time for this mini-batch computation
                # ===========================
                duration = time.time() - start_time

                # ===========================
                # write the summaries and print an overview fairly often
                # ===========================
                if (step+1) % exp_config.summary_writing_frequency == 0:                    
                    logging.info('Step %d: loss = %.3f (%.3f sec for the last step)' % (step+1, loss, duration))
                    
                    # ===========================
                    # Update the events file
                    # ===========================
                    summary_str = sess.run(summary, feed_dict = feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()

                # ===========================
                # Compute the loss on the entire training set
                # ===========================
                if (step+1) % exp_config.train_eval_frequency == 0:
                    logging.info('Training Data Eval:')
                    train_loss, train_dice = do_eval(sess,
                                                     eval_loss,
                                                     images_pl,
                                                     labels_pl,
                                                     training_pl,
                                                     imtr,
                                                     gttr,
                                                     exp_config.batch_size)                    
                    
                    tr_summary_msg = sess.run(tr_summary, feed_dict={tr_error: train_loss, tr_dice: train_dice})
                    summary_writer.add_summary(tr_summary_msg, step)
                    
                # ===========================
                # Save a checkpoint periodically
                # ===========================
                if step % exp_config.save_frequency == 0:
                    checkpoint_file = os.path.join(log_dir_tl, 'models/model.ckpt')
                    saver_tl.save(sess, checkpoint_file, global_step=step)

                # ===========================
                # Evaluate the model periodically on a validation set 
                # ===========================
                if (step+1) % exp_config.val_eval_frequency == 0:
                    logging.info('Validation Data Eval:')
                    val_loss, val_dice = do_eval(sess,
                                                 eval_loss,
                                                 images_pl,
                                                 labels_pl,
                                                 training_pl,
                                                 imvl,
                                                 gtvl,
                                                 exp_config.batch_size)                    
                    
                    vl_summary_msg = sess.run(vl_summary, feed_dict={vl_error: val_loss, vl_dice: val_dice})
                    summary_writer.add_summary(vl_summary_msg, step)

                    # ===========================
                    # save model if the val dice is the best yet
                    # ===========================
                    if val_dice > best_dice:
                        best_dice = val_dice
                        best_file = os.path.join(log_dir_tl, 'models/best_dice.ckpt')
                        saver_tl_best.save(sess, best_file, global_step=step)
                        logging.info('Found new average best dice on validation sets! - %f -  Saving model.' % val_dice)

                step += 1
                
        sess.close()

# ==================================================================
# ==================================================================
def do_eval(sess,
            eval_loss,
            images_placeholder,
            labels_placeholder,
            training_time_placeholder,
            images,
            labels,
            batch_size):

    loss_ii = 0
    dice_ii = 0
    num_batches = 0

    for batch in iterate_minibatches(images,
                                     labels,
                                     batch_size,
                                     train_or_eval = 'eval'):

        x, y = batch

        if y.shape[0] < batch_size:
            continue
        
        feed_dict = {images_placeholder: x, labels_placeholder: y, training_time_placeholder: False}
        loss, fg_dice = sess.run(eval_loss, feed_dict=feed_dict)
        
        loss_ii += loss
        dice_ii += fg_dice
        num_batches += 1

    avg_loss = loss_ii / num_batches
    avg_dice = dice_ii / num_batches

    logging.info('  Average segmentation loss: %.4f, average dice: %.4f' % (avg_loss, avg_dice))

    return avg_loss, avg_dice

# ==================================================================
# ==================================================================
def iterate_minibatches(images,
                        labels,
                        batch_size,
                        train_or_eval = 'train'):

    # ===========================
    # generate indices to randomly select subjects in each minibatch
    # ===========================
    n_images = images.shape[0]
    random_indices = np.random.permutation(n_images)

    # ===========================
    for b_i in range(n_images // batch_size):

        if b_i + batch_size > n_images:
            continue
        
        batch_indices = np.sort(random_indices[b_i*batch_size:(b_i+1)*batch_size])
        
        x = images[batch_indices, ...]
        y = labels[batch_indices, ...]

        # ===========================    
        # data augmentation (contrast changes + random elastic deformations)
        # ===========================      
        if exp_config.da_ratio > 0:

            # ===========================    
            # doing data aug both during training as well as during evaluation on the validation set (used for model selection)
            # ===========================             
            do_rot90 = args.train_dataset in ['HVHD', 'CSF', 'UHE']
            x, y = utils.do_data_augmentation(images = x,
                                              labels = y,
                                              data_aug_ratio = exp_config.da_ratio,
                                              sigma = exp_config.sigma,
                                              alpha = exp_config.alpha,
                                              trans_min = exp_config.trans_min,
                                              trans_max = exp_config.trans_max,
                                              rot_min = exp_config.rot_min,
                                              rot_max = exp_config.rot_max,
                                              scale_min = exp_config.scale_min,
                                              scale_max = exp_config.scale_max,
                                              gamma_min = exp_config.gamma_min,
                                              gamma_max = exp_config.gamma_max,
                                              brightness_min = exp_config.brightness_min,
                                              brightness_max = exp_config.brightness_max,
                                              noise_min = exp_config.noise_min,
                                              noise_max = exp_config.noise_max,
                                              rot90 = do_rot90)

        x = np.expand_dims(x, axis=-1)
        
        yield x, y

# ==================================================================
# ==================================================================
def main():
    
    # ===========================
    # Create dir if it does not exist
    # ===========================
    if not tf.gfile.Exists(log_dir_tl):
        tf.gfile.MakeDirs(log_dir_tl)
        tf.gfile.MakeDirs(log_dir_tl + '/models')
        tf.gfile.MakeDirs(tensorboard_dir_tl)

    if not tf.gfile.Exists(tensorboard_dir_tl):
        tf.gfile.MakeDirs(tensorboard_dir_tl)

    # run transfer learning
    run_transfer()

# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()
