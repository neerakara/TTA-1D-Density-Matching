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
import config.system as sys_config

import data.data_hcp as data_hcp
import data.data_abide as data_abide
import data.data_nci as data_nci
import data.data_promise as data_promise
import data.data_pirad_erc as data_pirad_erc

# ==================================================================
# setup logging
# ==================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments import i2i as exp_config

target_resolution = exp_config.target_resolution
image_size = exp_config.image_size
nlabels = exp_config.nlabels

log_dir = os.path.join(sys_config.project_root, 'log_dir/' + exp_config.expname_i2l)
logging.info('SD training directory: %s' %log_dir)
tensorboard_dir = os.path.join(sys_config.tensorboard_root, exp_config.expname_i2l)

# ================================================================
# load training data
# ================================================================
if exp_config.train_dataset is 'NCI':
    logging.info('Reading NCI images...')    
    logging.info('Data root directory: ' + sys_config.orig_data_root_nci)
    data_pros = data_nci.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_nci,
                                                        preprocessing_folder = sys_config.preproc_folder_nci,
                                                        size = image_size,
                                                        target_resolution = target_resolution,
                                                        force_overwrite = False,
                                                        cv_fold_num = 1)
    
    imtr, gttr = [ data_pros['images_train'], data_pros['masks_train'] ]
    imvl, gtvl = [ data_pros['images_validation'], data_pros['masks_validation'] ]

# ================================================================
# load test data
# ================================================================
if exp_config.test_dataset is 'PROMISE':
    data_pros = data_promise.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_promise,
                                                            preprocessing_folder = sys_config.preproc_folder_promise,
                                                            size = exp_config.image_size,
                                                            target_resolution = exp_config.target_resolution,
                                                            force_overwrite = False,
                                                            cv_fold_num = 2)
    
    imts = data_pros['images_test']
    gtts = data_pros['masks_test']
    name_test_subjects = data_pros['patnames_test']
    
    orig_data_res_x = data_pros['px_test'][:]
    orig_data_res_y = data_pros['py_test'][:]
    orig_data_res_z = data_pros['pz_test'][:]
    orig_data_siz_x = data_pros['nx_test'][:]
    orig_data_siz_y = data_pros['ny_test'][:]
    orig_data_siz_z = data_pros['nz_test'][:]

    num_test_subjects = orig_data_siz_z.shape[0] 
    ids = np.arange(num_test_subjects)

    logging.info(name_test_subjects)

# ================================================================
# Get patient IDs in the test set that belong to the UCL sub-dataset
# ================================================================
# sub_nums_ucl = []
# for sub_num in range(num_test_subjects):
#     if str(name_test_subjects[sub_num])[2:-1][4:6] in ['46']: # ['01', '26', '29', '31', '34', '36', '04', '06', '09', '39', '41']:
#         sub_nums_ucl.append(sub_num)

# for sub_num in sub_nums_ucl:
for sub_num in range(1,2):# (num_test_subjects):
    
    logging.info(str(name_test_subjects[sub_num])[2:-1])

    subject_name = str(name_test_subjects[sub_num])[2:-1]
    subject_string = exp_config.test_dataset + '_' + subject_name
    log_dir_tta = log_dir + exp_config.tta_string + subject_string
    tensorboard_dir_tta = tensorboard_dir  + exp_config.tta_string + exp_config.test_dataset + '_' + str(name_test_subjects[sub_num])[2:-1]
    logging.info('Tensorboard directory TTA: %s' %tensorboard_dir_tta)

    if not tf.gfile.Exists(log_dir_tta):
        tf.gfile.MakeDirs(log_dir_tta)
        tf.gfile.MakeDirs(tensorboard_dir_tta)
    
    subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])
    subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])
    test_image = imts[subject_id_start_slice:subject_id_end_slice,:,:]  
    test_image_gt = gtts[subject_id_start_slice:subject_id_end_slice,:,:]  
    test_image_gt = test_image_gt.astype(np.uint8)

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
                                                training_pl = tf.constant(False, dtype=tf.bool))
                        
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

        tta_vars = normalization_vars # bn_vars # normalization_vars

        # ================================================================
        # Define BN stats matching loss
        # ================================================================
        # placeholders for SD stats. These will be extracted after loading the SD trained model.
        sd_mu_pl = tf.placeholder(tf.float32, shape = [None], name = 'sd_means')
        sd_var_pl = tf.placeholder(tf.float32, shape = [None], name = 'sd_variances')

        # compute the stats of features of the TD image that is fed via the placeholder
        td_means = tf.zeros([1])
        td_variances = tf.ones([1])
        for conv_block in [1,2,3,4,5,6,7]:
            for conv_sub_block in [1,2]:
                conv_string = str(conv_block) + '_' + str(conv_sub_block)
                features = tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '/Conv2D:0')
                this_layer_means, this_layer_variances = tf.nn.moments(features, axes = [0,1,2])
                td_means = tf.concat([td_means, this_layer_means], 0)
                td_variances = tf.concat([td_variances, this_layer_variances], 0)
        td_mu = td_means[1:]
        td_var = td_variances[1:]

        # compute the TTA loss
        loss_op = tf.math.log(td_var / sd_var_pl) + (sd_var_pl + (sd_mu_pl - td_mu)**2) / td_var
        loss_op = tf.reduce_mean(loss_op) # mean over all channels of all layers

        # ================================================================
        # add losses to tensorboard
        # ================================================================
        tf.summary.scalar('loss_tta_kl', loss_op)         
        summary_during_tta = tf.summary.merge_all()
        
        # ================================================================
        # add optimization ops
        # ================================================================   
        # create an instance of the required optimizer
        optimizer = exp_config.optimizer_handle(learning_rate = exp_config.learning_rate)    
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
        display_histograms_pl = tf.placeholder(tf.uint8, shape = [1, None, None, 1], name='display_histograms_pl')
        histograms_summary = tf.summary.image('histograms', display_histograms_pl)

        # ================================================================
        # create saver
        # ================================================================
        saver_i2l = tf.train.Saver(var_list = i2l_vars)
        saver_normalizer = tf.train.Saver(var_list = normalization_vars)      
        saver_normalizer_best = tf.train.Saver(var_list = normalization_vars, max_to_keep=3)   
                
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
        path_to_model = sys_config.project_root + 'log_dir/' + exp_config.expname_i2l + 'models/'
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_dice.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_i2l.restore(sess, checkpoint_path)
        
        # ================================================================
        # extract the SD stats once, and pass them as placeholders for computing the loss in each iteration
        # ================================================================
        sd_mu = np.array([])
        sd_var = np.array([])    
        for conv_block in [1,2,3,4,5,6,7]:
            for conv_sub_block in [1,2]:
                conv_string = str(conv_block) + '_' + str(conv_sub_block)
                tmp_mu = sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '_bn/moving_mean:0'))
                tmp_var = sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '_bn/moving_variance:0'))
                sd_mu = np.concatenate([sd_mu, tmp_mu])
                sd_var = np.concatenate([sd_var, tmp_var])

        step = 0
        best_loss = 1000.0

        # ================================================================
        # TTA iterations
        # ================================================================
        while (step < exp_config.max_steps_i2i):
            # run accumulated_gradients_zero_op (no need to provide values for any placeholders)
            sess.run(accumulated_gradients_zero_op)
            num_accumulation_steps = 0
            # run the accumulate gradients op             
            sess.run(accumulate_gradients_op, feed_dict={images_pl: np.expand_dims(test_image, axis=-1),
                                                        sd_mu_pl: sd_mu, 
                                                        sd_var_pl: sd_var})
            num_accumulation_steps = num_accumulation_steps + 1
            # run accumulated_gradients_mean_op, with a value for the placeholder num_accumulation_steps_pl
            sess.run(accumulated_gradients_mean_op, feed_dict = {num_accumulation_steps_pl: num_accumulation_steps})
            # run the train_op. this also requires input output placeholders, as compute_gradients will be called again, but the returned gradient values will be replaced by the mean gradients.
            sess.run(train_op, feed_dict={images_pl: np.expand_dims(test_image, axis=-1), sd_mu_pl: sd_mu,  sd_var_pl: sd_var})

            # ===========================
            # get dice wrt ground truth
            # ===========================
            label_predicted = sess.run(preds, feed_dict = {images_pl: np.expand_dims(test_image, axis=-1)})
            label_predicted = np.squeeze(np.array(label_predicted)).astype(float)  
            if exp_config.test_dataset is 'PROMISE':
                label_predicted[label_predicted!=0] = 1
            dice_wrt_gt = met.f1_score(test_image_gt.flatten(), label_predicted.flatten(), average=None) 
            summary_writer.add_summary(sess.run(gt_dice_summary, feed_dict={gt_dice: np.mean(dice_wrt_gt[1:])}), step)

            # ===========================
            # Update the events file
            # ===========================
            loss_this_step = sess.run(loss_op, feed_dict={images_pl: np.expand_dims(test_image, axis=-1), sd_mu_pl: sd_mu,  sd_var_pl: sd_var})
            summary_str = sess.run(summary_during_tta, feed_dict = {images_pl: np.expand_dims(test_image, axis=-1), sd_mu_pl: sd_mu,  sd_var_pl: sd_var})
            summary_writer.add_summary(summary_str, step)
            summary_writer.flush()

            if step % 10 == 0:
                # ===========================
                # get normalized image
                # ===========================
                image_normalized = sess.run(images_normalized, feed_dict = {images_pl: np.expand_dims(test_image, axis=-1)})        
                image_normalized = np.squeeze(np.array(image_normalized)).astype(float)  

                # ===========================   
                # visualize images
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
                # visualize feature distribution alignment
                # ===========================
                td_mu_this_step, td_var_this_step = sess.run([td_mu, td_var], feed_dict = {images_pl: np.expand_dims(test_image, axis=-1)})       
                utils_vis.write_histogram_summaries(step,
                                                    summary_writer,
                                                    sess,
                                                    histograms_summary,
                                                    display_histograms_pl,
                                                    sd_mu,
                                                    sd_var,
                                                    td_mu_this_step,
                                                    td_var_this_step,
                                                    log_dir_tta)

            # ===========================
            # save best model so far
            # ===========================
            if best_loss > loss_this_step:
                best_loss = loss_this_step
                best_file = os.path.join(log_dir_tta, 'models/best_loss.ckpt')
                saver_normalizer_best.save(sess, best_file, global_step=step)
                logging.info('Found new best score (%f) at step %d -  Saving model.' % (best_loss, step))

            step = step + 1


        # ================================================================================================================================
        # Initial code to print SD and TD stats
        # ================================================================================================================================

        # ================================================================
        # figure out how to access BN stats
        # ================================================================
        # logging.info("These are the SD BN stats stored in the saved model.")
        # for v in bn_stats:
        #     logging.info(v.name)

        # ================================================================
        # Print op names to figure out how exactly to run the required things
        # ================================================================
        # for op in tf.get_default_graph().get_operations():
        #     if 'conv1_1' in op.name:
        #         logging.info(op.name)
            # i2l_mapper/conv1_1/kernel/Initializer/random_uniform/shape
            # i2l_mapper/conv1_1/kernel/Initializer/random_uniform/min
            # i2l_mapper/conv1_1/kernel/Initializer/random_uniform/max
            # i2l_mapper/conv1_1/kernel/Initializer/random_uniform/RandomUniform
            # i2l_mapper/conv1_1/kernel/Initializer/random_uniform/sub
            # i2l_mapper/conv1_1/kernel/Initializer/random_uniform/mul
            # i2l_mapper/conv1_1/kernel/Initializer/random_uniform
            # i2l_mapper/conv1_1/kernel
            # i2l_mapper/conv1_1/kernel/Assign
            # i2l_mapper/conv1_1/kernel/read

            # i2l_mapper/conv1_1/dilation_rate
            # i2l_mapper/conv1_1/Conv2D <--------- THE FEATURES BEFORE THE BN LAYER

            # i2l_mapper/conv1_1_bn/gamma/Initializer/ones
            # i2l_mapper/conv1_1_bn/gamma <--------- GAMMA VALUE LEARNED ON SD
            # i2l_mapper/conv1_1_bn/gamma/Assign
            # i2l_mapper/conv1_1_bn/gamma/read
            # i2l_mapper/conv1_1_bn/beta/Initializer/zeros
            # i2l_mapper/conv1_1_bn/beta <--------- BETA VALUE LEARNED ON SD
            # i2l_mapper/conv1_1_bn/beta/Assign
            # i2l_mapper/conv1_1_bn/beta/read
            # i2l_mapper/conv1_1_bn/moving_mean/Initializer/zeros
            # i2l_mapper/conv1_1_bn/moving_mean <--------- MEAN OF SD
            # i2l_mapper/conv1_1_bn/moving_mean/Assign
            # i2l_mapper/conv1_1_bn/moving_mean/read
            # i2l_mapper/conv1_1_bn/moving_variance/Initializer/ones
            # i2l_mapper/conv1_1_bn/moving_variance <--------- VARIANCE OF SD
            # i2l_mapper/conv1_1_bn/moving_variance/Assign
            # i2l_mapper/conv1_1_bn/moving_variance/read
            # i2l_mapper/conv1_1_bn/FusedBatchNorm <--------- THE FEATURES AFTER THE BN LAYER?
            # i2l_mapper/conv1_1_bn/Const <--------- THE FEATURES AFTER THE BN LAYER?

            # i2l_mapper/Relu <--------- THE FEATURES AFTER THE BN LAYER, FOLLOWED BY ACTIVATION FUNCTION.

        
        # # ================================================================
        # # plot / print SD stats stored in the BN layers of the saved model
        # # ================================================================
        # # get the tensor at the output of an operation by adding ':0' at the end of the op name
        # # conv1_1
        # sd_mu = sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv1_1_bn/moving_mean:0'))
        # sd_var = sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv1_1_bn/moving_variance:0'))
        # sd_gamma = sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv1_1_bn/gamma:0'))
        # sd_beta = sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv1_1_bn/beta:0'))
        # # logging.info(sd_mu.shape) # [num_channels]
        # # logging.info(sd_mu)
        # # logging.info(sd_var.shape) # [num_channels]
        # # logging.info(sd_var)    
        # # logging.info(sd_gamma.shape) # [num_channels]
        # # logging.info(sd_gamma)    
        # # logging.info(sd_beta.shape) # [num_channels]
        # # logging.info(sd_beta)    

        # # ================================================================
        # # plot / print stats of the test image
        # # ================================================================
        # # For Prostate, we have around 20-30 slices, so we can probably feed the entire 3D image at once without running into memory problems
        # td_features = sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv1_1/Conv2D:0'),
        #                        feed_dict={images_pl: np.expand_dims(test_image, axis=-1)})
        # # logging.info(td_features.shape) 
        # ti_mu = td_features.mean(axis=(0, 1, 2)) # do it in tf
        # ti_var = td_features.var(axis=(0, 1, 2)) # do it in tf
        # # logging.info(ti_mu.shape) # [num_channels]
        # # logging.info(ti_mu)    
        # # logging.info(ti_var.shape) # [num_channels]
        # # logging.info(ti_var)    

        # # ================================================================
        # # Compute KL divergence between the per-channel 1D distributions, then average over the channels. This is the TTA loss.
        # # https://github.com/neerakara/Domain-Shift-Literature/blob/main/notes/bn_adabn.md
        # # ================================================================
        # loss_kl = np.log(ti_var / sd_var) + (sd_var + (sd_mu - ti_mu)**2) / ti_var
        # # logging.info(loss_kl.shape) # [num_channels]
        # # logging.info(loss_kl)    

        # loss_kl = np.mean(loss_kl)
        # # logging.info(loss_kl.shape) # [1]
        # logging.info("Loss considering only the given test image: " + str(loss_kl))

        # # ================================================================
        # # Compute loss for SD+DA vs TI+DA
        # # ================================================================
        # p = 0.9 # momemtum
        # for _ in range(100):
        #     ti_aug, _ = utils.do_data_augmentation(images = test_image,
        #                                            labels = test_image,
        #                                            data_aug_ratio = exp_config.da_ratio,
        #                                            sigma = exp_config.sigma,
        #                                            alpha = exp_config.alpha,
        #                                            trans_min = exp_config.trans_min,
        #                                            trans_max = exp_config.trans_max,
        #                                            rot_min = exp_config.rot_min,
        #                                            rot_max = exp_config.rot_max,
        #                                            scale_min = exp_config.scale_min,
        #                                            scale_max = exp_config.scale_max,
        #                                            gamma_min = exp_config.gamma_min,
        #                                            gamma_max = exp_config.gamma_max,
        #                                            brightness_min = exp_config.brightness_min,
        #                                            brightness_max = exp_config.brightness_max,
        #                                            noise_min = exp_config.noise_min,
        #                                            noise_max = exp_config.noise_max)

        #     td_features = sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv1_1/Conv2D:0'),
        #                            feed_dict={images_pl: np.expand_dims(ti_aug, axis=-1)})
            
        #     ti_mu = p*ti_mu + (1-p)*td_features.mean(axis=(0, 1, 2)) # do it in tf
        #     ti_var = p*ti_var + (1-p)*td_features.var(axis=(0, 1, 2)) # do it in tf
            
        # loss_kl = np.mean(np.log(ti_var / sd_var) + (sd_var + (sd_mu - ti_mu)**2) / ti_var)
        # logging.info("Loss considering the given test image and its augmented versions: " + str(loss_kl))

        # ================================================================
        # close session
        # ================================================================
        sess.close()