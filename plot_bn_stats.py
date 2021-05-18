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
ucl_ids = ['01', '26', '29', '31', '34', '36']
hk_ids = ['39', '41', '44', '46', '49']
sub_nums_ucl = []
for sub_num in range(num_test_subjects):
    if str(name_test_subjects[sub_num])[2:-1][4:6] in ucl_ids:
        sub_nums_ucl.append(sub_num)

sub_num = sub_nums_ucl[0]
subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])
subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])
test_image = imts[subject_id_start_slice:subject_id_end_slice,:,:]  
logging.info(str(name_test_subjects[sub_num])[2:-1])

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
    bn_stats = []    
    for v in tf.global_variables():
        var_name = v.name        
        i2l_vars.append(v) # important to store all global variables and not just trainable variables here (so that BN stats are also restored)
        if 'image_normalizer' in var_name:
            normalization_vars.append(v)
        if 'beta' in var_name or 'gamma' in var_name:
            bn_stats.append(v)
            logging.info(var_name)

                            
    # ================================================================
    # add init ops
    # ================================================================
    init_ops = tf.global_variables_initializer()
            
    # ================================================================
    # create session
    # ================================================================
    sess = tf.Session()

    # ================================================================
    # create saver
    # ================================================================
    saver_i2l = tf.train.Saver(var_list = i2l_vars)
    saver_normalizer = tf.train.Saver(var_list = normalization_vars)        
            
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
    # Restore the normalization network parameters
    # ================================================================
    if exp_config.normalize is True:
        logging.info('============================================================')
        path_to_model = sys_config.project_root + 'log_dir/' + exp_config.expname_normalizer + 'subject_' + subject_name + '/models/'
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_loss.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_normalizer.restore(sess, checkpoint_path)
        logging.info('============================================================')

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

    
    # ================================================================
    # plot / print SD stats stored in the BN layers of the saved model
    # ================================================================
    # get the tensor at the output of an operation by adding ':0' at the end of the op name
    # conv1_1
    sd_mu = sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv1_1_bn/moving_mean:0'))
    sd_var = sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv1_1_bn/moving_variance:0'))
    sd_gamma = sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv1_1_bn/gamma:0'))
    sd_beta = sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv1_1_bn/beta:0'))
    # logging.info(sd_mu.shape) # [num_channels]
    # logging.info(sd_mu)
    # logging.info(sd_var.shape) # [num_channels]
    # logging.info(sd_var)    
    # logging.info(sd_gamma.shape) # [num_channels]
    # logging.info(sd_gamma)    
    # logging.info(sd_beta.shape) # [num_channels]
    # logging.info(sd_beta)    

    # ================================================================
    # plot / print stats of the test image
    # ================================================================
    # For Prostate, we have around 20-30 slices, so we can probably feed the entire 3D image at once without running into memory problems
    td_features = sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv1_1/Conv2D:0'),
                           feed_dict={images_pl: np.expand_dims(test_image, axis=-1)})
    # logging.info(td_features.shape) 
    ti_mu = td_features.mean(axis=(0, 1, 2)) # do it in tf
    ti_var = td_features.var(axis=(0, 1, 2)) # do it in tf
    # logging.info(ti_mu.shape) # [num_channels]
    # logging.info(ti_mu)    
    # logging.info(ti_var.shape) # [num_channels]
    # logging.info(ti_var)    

    # ================================================================
    # Compute KL divergence between the per-channel 1D distributions, then average over the channels. This is the TTA loss.
    # https://github.com/neerakara/Domain-Shift-Literature/blob/main/notes/bn_adabn.md
    # ================================================================
    loss_kl = np.log(ti_var / sd_var) + (sd_var + (sd_mu - ti_mu)**2) / ti_var
    # logging.info(loss_kl.shape) # [num_channels]
    # logging.info(loss_kl)    

    loss_kl = np.mean(loss_kl)
    # logging.info(loss_kl.shape) # [1]
    logging.info("Loss considering only the given test image: " + str(loss_kl))

    # ================================================================
    # Compute loss for SD+DA vs TI+DA
    # ================================================================
    p = 0.9 # momemtum
    for _ in range(1):
        ti_aug, _ = utils.do_data_augmentation(images = test_image,
                                               labels = test_image,
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
                                               noise_max = exp_config.noise_max)

        td_features = sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv1_1/Conv2D:0'),
                               feed_dict={images_pl: np.expand_dims(ti_aug, axis=-1)})
        
        ti_mu = p*ti_mu + (1-p)*td_features.mean(axis=(0, 1, 2)) # do it in tf
        ti_var = p*ti_var + (1-p)*td_features.var(axis=(0, 1, 2)) # do it in tf
        
    loss_kl = np.mean(np.log(ti_var / sd_var) + (sd_var + (sd_mu - ti_mu)**2) / ti_var)
    logging.info("Loss considering the given test image and its augmented versions: " + str(loss_kl))

    # ================================================================
    # close session
    # ================================================================
    sess.close()