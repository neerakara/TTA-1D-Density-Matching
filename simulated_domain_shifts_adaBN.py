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
import data.data_acdc as data_acdc
import data.data_rvsc as data_rvsc

from skimage.transform import rescale
import sklearn.metrics as met

# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments import i2i as exp_config
log_root = sys_config.project_root + 'log_dir/'

log_dir_sd = log_root + exp_config.expname_i2l
log_dir_tta = log_dir_sd + exp_config.tta_string

target_resolution = exp_config.target_resolution
image_size = exp_config.image_size
nlabels = exp_config.nlabels

# ==================================================================
# main function for inference
# ==================================================================
def predict_segmentation(subject_name,
                         image,
                         normalize = True):
    
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
        
        for v in tf.global_variables():
            var_name = v.name        
            i2l_vars.append(v)
            if 'image_normalizer' in var_name:
                normalization_vars.append(v)

        # ================================================================
        # define ops to compute feature means and variances at each layer
        # ================================================================                 
        if exp_config.adaBN == True:               
            td_means = []
            td_variances = []
            mean_assign_ops = []
            variance_assign_ops = []
            tmp_mean_pl = tf.placeholder(tf.float32, shape = [None], name = 'tmp_mean')
            tmp_variance_pl = tf.placeholder(tf.float32, shape = [None], name = 'tmp_variance')
            for conv_block in [1,2,3,4,5,6,7]:
                for conv_sub_block in [1,2]:
                    conv_string = str(conv_block) + '_' + str(conv_sub_block)
                    features = tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '/Conv2D:0')
                    this_layer_means, this_layer_variances = tf.nn.moments(features, axes = [0,1,2])
                    td_means.append(this_layer_means)
                    td_variances.append(this_layer_variances)
                    mean_assign_ops.append(tf.assign(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '_bn/moving_mean:0'), tmp_mean_pl))
                    variance_assign_ops.append(tf.assign(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '_bn/moving_variance:0'), tmp_variance_pl))

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
        path_to_model = log_dir_sd + 'models/'
        checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_dice.ckpt')
        logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        saver_i2l.restore(sess, checkpoint_path)
        
        # ================================================================
        # Restore the normalization network parameters
        # ================================================================
        # if normalize == True:
        #     logging.info('============================================================')
        #     subject_string = exp_config.test_dataset + '_' + subject_name
        #     path_to_model = log_dir_tta + subject_string + '/models/'
        #     checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_loss.ckpt')
        #     logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
        #     saver_normalizer.restore(sess, checkpoint_path)
        #     logging.info('============================================================')

        # ================================================================
        # Compute mean and variance of the current 3D test image for features of each channel in each layer.
        # Replace the SD mean and SD variance stored in the BN layers with the computed values.
        # Do this one layer at a time - that is first replace the means and variances in the first BN layer.
        # Now, compute the means and variances of the TD features at the second layer and replace them.
        # Next, follow the same step for the next layers one by one.
        # ================================================================
        # This works for the Prostate dataset, but we cannot pass the whole image at once if the image is too big (eg brain). need to do this in batches
        # if exp_config.adaBN == True:
        #     count = 0
        #     for conv_block in [1,2,3,4,5,6,7]:
        #         for conv_sub_block in [1,2]:
        #             td_means_this_layer = sess.run(td_means, feed_dict={images_pl: np.expand_dims(image, axis=-1)})[count]
        #             td_variances_this_layer = sess.run(td_variances, feed_dict={images_pl: np.expand_dims(image, axis=-1)})[count]
        #             sess.run(mean_assign_ops[count], feed_dict={tmp_mean_pl: td_means_this_layer})
        #             sess.run(variance_assign_ops[count], feed_dict={tmp_variance_pl: td_variances_this_layer})                
        #             count = count + 1
        # ================================================================
        if exp_config.adaBN == True:
            count = 0
            for conv_block in [1,2,3,4,5,6,7]:
                for conv_sub_block in [1,2]:
                    
                    # divide the image into batches
                    b_size = 32
                    num_batches = 0
                    for b_i in range(0, image.shape[0], b_size):

                        if b_i + b_size < image.shape[0]:
                            batch = np.expand_dims(image[b_i:b_i+b_size, ...], axis=-1)
                        else:
                            batch = np.expand_dims(image[b_i:, ...], axis=-1)
                        
                        if b_i == 0:
                            td_means_this_layer = sess.run(td_means, feed_dict={images_pl: batch})[count]
                            td_variances_this_layer = sess.run(td_variances, feed_dict={images_pl: batch})[count]
                        else:
                            td_means_this_layer = td_means_this_layer + sess.run(td_means, feed_dict={images_pl: batch})[count]
                            td_variances_this_layer = td_variances_this_layer + sess.run(td_variances, feed_dict={images_pl: batch})[count]

                        num_batches = num_batches + 1

                    td_means_this_layer = td_means_this_layer / num_batches
                    td_variances_this_layer = td_variances_this_layer / num_batches

                    # =============
                    # conv_string = str(conv_block) + '_' + str(conv_sub_block)
                    # logging.info('Value before assignment')
                    # logging.info(sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '_bn/moving_mean:0'))[0])
                    # =============
                    # Assign these values to moving means and moving variances before doing the prediction
                    # =============
                    sess.run(mean_assign_ops[count], feed_dict={tmp_mean_pl: td_means_this_layer})
                    sess.run(variance_assign_ops[count], feed_dict={tmp_variance_pl: td_variances_this_layer})                
                    # =============
                    # logging.info('Value after assignment')
                    # logging.info(sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '_bn/moving_mean:0'))[0])
                    # logging.info('=============')
                    # =============
                    count = count + 1

        # ================================================================
        # Make predictions for the image at the resolution of the image after pre-processing
        # ================================================================
        label_predicted = []
        image_normalized = []
        
        for b_i in range(0, image.shape[0], 1):
        
            X = np.expand_dims(image[b_i:b_i+1, ...], axis=-1)
            
            label_predicted.append(sess.run(preds, feed_dict={images_pl: X}))
            image_normalized.append(sess.run(images_normalized, feed_dict={images_pl: X}))
        
        label_predicted = np.squeeze(np.array(label_predicted)).astype(float)  
        image_normalized = np.squeeze(np.array(image_normalized)).astype(float)  
        
        sess.close()
        
        return label_predicted, image_normalized
    
# ================================================================
# ================================================================
def rescale_and_crop(arr,
                     px,
                     py,
                     nx,
                     ny,
                     order_interpolation,
                     num_rotations):
    
    # 'target_resolution_brain' contains the resolution that the images were rescaled to, during the pre-processing.
    # we need to undo this rescaling before evaluation
    scale_vector = [target_resolution[0] / px,
                    target_resolution[1] / py]

    arr_list = []
    
    for zz in range(arr.shape[0]):
     
        # ============
        # rotate the labels back to the original orientation
        # ============            
        arr2d_rotated = np.rot90(np.squeeze(arr[zz, :, :]), k=num_rotations)
        
        arr2d_rescaled = rescale(arr2d_rotated,
                                 scale_vector,
                                 order = order_interpolation,
                                 preserve_range = True,
                                 multichannel = False,
                                 anti_aliasing = False,
                                 mode = 'constant')

        arr2d_rescaled_cropped = utils.crop_or_pad_slice_to_size(arr2d_rescaled, nx, ny)

        arr_list.append(arr2d_rescaled_cropped)
    
    arr_orig_res_and_size = np.array(arr_list)
    arr_orig_res_and_size = arr_orig_res_and_size.swapaxes(0, 1).swapaxes(1, 2)
    
    return arr_orig_res_and_size
        
# ==================================================================
# ==================================================================
def simulate_ds(image, ds_type, ds_params, normalize_after_ds):
    
    if ds_type == 'gamma':
        image_tmp = image**ds_params[0]
    elif ds_type == 'linear':
        image_tmp = ds_params[0]*image + ds_params[1]
    elif ds_type == 'quadratic':
        image_tmp = ds_params[0]*(image**2) + ds_params[1]*image + ds_params[2]
    elif ds_type == 'fourth_order':
        image_tmp = ds_params[0]*(image**4) + ds_params[1]*(image**3) + ds_params[2]*(image**2) + ds_params[3]*(image) + ds_params[4]
    if normalize_after_ds == True:
        image_tmp = utils.normalise_image(image_tmp)

    return image_tmp

# ==================================================================
# ==================================================================
def main():
    
    # ===================================
    # read the test images
    # ===================================
    test_dataset_name = exp_config.test_dataset
    
    if test_dataset_name == 'HCPT1':
        logging.info('Reading HCPT1 images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_hcp)
        
        image_depth = exp_config.image_depth_hcp
        idx_start = 50
        idx_end = 70       
        
        data_brain_test = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                               preprocessing_folder = sys_config.preproc_folder_hcp,
                                                               idx_start = idx_start,
                                                               idx_end = idx_end,                
                                                               protocol = 'T1',
                                                               size = image_size,
                                                               depth = image_depth,
                                                               target_resolution = target_resolution)

        imts = data_brain_test['images']
        name_test_subjects = data_brain_test['patnames']
        num_test_subjects = imts.shape[0] // image_depth
        ids = np.arange(idx_start, idx_end)

        orig_data_res_x = data_brain_test['px'][:]
        orig_data_res_y = data_brain_test['py'][:]
        orig_data_res_z = data_brain_test['pz'][:]
        orig_data_siz_x = data_brain_test['nx'][:]
        orig_data_siz_y = data_brain_test['ny'][:]
        orig_data_siz_z = data_brain_test['nz'][:]
        
    elif test_dataset_name == 'HCPT2':
        logging.info('Reading HCPT2 images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_hcp)
        
        image_depth = exp_config.image_depth_hcp
        idx_start = 50
        idx_end = 70
        
        data_brain_test = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                               preprocessing_folder = sys_config.preproc_folder_hcp,
                                                               idx_start = idx_start,
                                                               idx_end = idx_end,           
                                                               protocol = 'T2',
                                                               size = image_size,
                                                               depth = image_depth,
                                                               target_resolution = target_resolution)

        imts = data_brain_test['images']
        name_test_subjects = data_brain_test['patnames']
        num_test_subjects = imts.shape[0] // image_depth
        ids = np.arange(idx_start, idx_end)

        orig_data_res_x = data_brain_test['px'][:]
        orig_data_res_y = data_brain_test['py'][:]
        orig_data_res_z = data_brain_test['pz'][:]
        orig_data_siz_x = data_brain_test['nx'][:]
        orig_data_siz_y = data_brain_test['ny'][:]
        orig_data_siz_z = data_brain_test['nz'][:]
        
    elif test_dataset_name == 'CALTECH':
        logging.info('Reading CALTECH images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_abide + 'CALTECH/')
        
        image_depth = exp_config.image_depth_caltech
        idx_start = 16
        idx_end = 36         
        
        data_brain_test = data_abide.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_abide,
                                                                 preprocessing_folder = sys_config.preproc_folder_abide,
                                                                 site_name = 'CALTECH',
                                                                 idx_start = idx_start,
                                                                 idx_end = idx_end,             
                                                                 protocol = 'T1',
                                                                 size = image_size,
                                                                 depth = image_depth,
                                                                 target_resolution = target_resolution)

        imts = data_brain_test['images']
        name_test_subjects = data_brain_test['patnames']
        num_test_subjects = imts.shape[0] // image_depth
        ids = np.arange(idx_start, idx_end)

        orig_data_res_x = data_brain_test['px'][:]
        orig_data_res_y = data_brain_test['py'][:]
        orig_data_res_z = data_brain_test['pz'][:]
        orig_data_siz_x = data_brain_test['nx'][:]
        orig_data_siz_y = data_brain_test['ny'][:]
        orig_data_siz_z = data_brain_test['nz'][:]

    elif test_dataset_name == 'STANFORD':
        logging.info('Reading STANFORD images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_abide + 'STANFORD/')
        
        image_depth = exp_config.image_depth_stanford
        idx_start = 16
        idx_end = 36         
        
        data_brain_test = data_abide.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_abide,
                                                                 preprocessing_folder = sys_config.preproc_folder_abide,
                                                                 site_name = 'STANFORD',
                                                                 idx_start = idx_start,
                                                                 idx_end = idx_end,             
                                                                 protocol = 'T1',
                                                                 size = image_size,
                                                                 depth = image_depth,
                                                                 target_resolution = target_resolution)

        imts = data_brain_test['images']
        name_test_subjects = data_brain_test['patnames']
        num_test_subjects = imts.shape[0] // image_depth
        ids = np.arange(idx_start, idx_end)

        orig_data_res_x = data_brain_test['px'][:]
        orig_data_res_y = data_brain_test['py'][:]
        orig_data_res_z = data_brain_test['pz'][:]
        orig_data_siz_x = data_brain_test['nx'][:]
        orig_data_siz_y = data_brain_test['ny'][:]
        orig_data_siz_z = data_brain_test['nz'][:]

    elif test_dataset_name == 'NCI':
        data_pros = data_nci.load_and_maybe_process_data(input_folder=sys_config.orig_data_root_nci,
                                                         preprocessing_folder=sys_config.preproc_folder_nci,
                                                         size=image_size,
                                                         target_resolution=target_resolution,
                                                         force_overwrite=False,
                                                         cv_fold_num = 1)

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

    elif test_dataset_name == 'PIRAD_ERC':

        idx_start = 0
        idx_end = 20
        ids = np.arange(idx_start, idx_end)

        data_pros = data_pirad_erc.load_data(input_folder=sys_config.orig_data_root_pirad_erc,
                                             preproc_folder=sys_config.preproc_folder_pirad_erc,
                                             idx_start=idx_start,
                                             idx_end=idx_end,
                                             size=image_size,
                                             target_resolution=target_resolution,
                                             labeller='ek')
        imts = data_pros['images']
        name_test_subjects = data_pros['patnames']

        orig_data_res_x = data_pros['px'][:]
        orig_data_res_y = data_pros['py'][:]
        orig_data_res_z = data_pros['pz'][:]
        orig_data_siz_x = data_pros['nx'][:]
        orig_data_siz_y = data_pros['ny'][:]
        orig_data_siz_z = data_pros['nz'][:]

        num_test_subjects = orig_data_siz_z.shape[0]

    elif test_dataset_name == 'PROMISE':
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
            
    
    # ==================================================================
    # define hyperparameters of the simulated domain shift - so that the results can be named accordingly
    # ==================================================================
    ds = True
    ds_type = 'gamma' # 'gamma', 'linear' 
    normalize_after_ds = True
    for ds_param1 in [4.9]: # gamma : [0.1, 0.25, 0.5, 1.0, 1.25, 1.5, 2.5, 5.0], linear_a: [1.0, 10.0, -10.0]
        ds_params = []
        if ds == True:
            ds_params.append(ds_param1)
            if ds_type == 'gamma':
                ds_str = '_ds_' + str(ds_type) + '_' + str(ds_param1)
            elif ds_type == 'linear':
                ds_param2 = 50.0
                ds_params.append(ds_param2)
                ds_str = '_ds_' + str(ds_type) + '_' + str(ds_param1) + '_' + str(ds_param2)
            elif ds_type == 'quadratic':
                ds_param2 = 1.0
                ds_param3 = 0.0
                ds_params.append(ds_param2)
                ds_params.append(ds_param3)
                ds_str = '_ds_' + str(ds_type) + '_' + str(ds_param1) + '_' + str(ds_param2) + '_' + str(ds_param3)
            elif ds_type == 'fourth_order':
                ds_param2 = 10.0
                ds_param3 = -11.0
                ds_param4 = 4.0
                ds_param5 = 0.0
                ds_params.append(ds_param2)
                ds_params.append(ds_param3)
                ds_params.append(ds_param4)
                ds_params.append(ds_param5)
                ds_str = '_ds_' + str(ds_type) + '_' + str(ds_param1) + '_' + str(ds_param2) + '_' + str(ds_param3) + '_' + str(ds_param4) + '_' + str(ds_param5)
            if normalize_after_ds == True:
                ds_str = ds_str + '_normTrue'
            else:
                ds_str = ds_str + '_normFalse'
        else:
            ds_str = ''

        # ================================   
        # open a text file for writing the mean dice scores for each subject that is evaluated
        # ================================    
        if exp_config.normalize == True:
            if not tf.gfile.Exists(log_dir_tta):
                tf.gfile.MakeDirs(log_dir_tta)
            log_dir = log_dir_tta
        else:
            log_dir = log_dir_sd
        results_prefix = log_dir + test_dataset_name + '_whole' + str(int(exp_config.whole_gland_results)) + '_adaBN' + str(int(exp_config.adaBN)) + ds_str
        results_file = open(results_prefix + '.txt', "w")
        results_file.write("================================== \n") 
        results_file.write("Test results \n") 
        
        # ================================================================
        # For each test image, load the best model and compute the dice with this model
        # ================================================================
        dice_per_label_per_subject = []
        hsd_per_label_per_subject = []

        # for now, select a subject in the PROMISE dataset that gives good performance after training with the NCI dataset
        # We can consider this to come from the same domain as NCI (it actually does come from the same sub-dataset)
        for sub_num in range(4,5):#(num_test_subjects):

            subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])
            subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])
            image = imts[subject_id_start_slice:subject_id_end_slice,:,:]  
            
            if ds == True:
                image = simulate_ds(image, ds_type, ds_params, normalize_after_ds)

            # ==================================================================
            # setup logging
            # ==================================================================
            logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
            subject_name = str(name_test_subjects[sub_num])[2:-1]
            logging.info('============================================================')
            logging.info('Subject id: %s' %sub_num)
        
            # ==================================================================
            # predict segmentation at the pre-processed resolution
            # ==================================================================
            predicted_labels, normalized_image = predict_segmentation(subject_name,
                                                                    image,
                                                                    exp_config.normalize)
                    
            # ==================================================================
            # read the original segmentation mask
            # ==================================================================
            if test_dataset_name is 'HCPT1':
                # image will be normalized to [0,1]
                image_orig, labels_orig = data_hcp.load_without_size_preprocessing(input_folder = sys_config.orig_data_root_hcp,
                                                                                idx = ids[sub_num],
                                                                                protocol = 'T1',
                                                                                preprocessing_folder = sys_config.preproc_folder_hcp,
                                                                                depth = image_depth)
                num_rotations = 0  
                
            elif test_dataset_name is 'HCPT2':
                # image will be normalized to [0,1]
                image_orig, labels_orig = data_hcp.load_without_size_preprocessing(input_folder = sys_config.orig_data_root_hcp,
                                                                                idx = ids[sub_num],
                                                                                protocol = 'T2',
                                                                                preprocessing_folder = sys_config.preproc_folder_hcp,
                                                                                depth = image_depth)
                num_rotations = 0  

            elif test_dataset_name is 'CALTECH':
                # image will be normalized to [0,1]
                image_orig, labels_orig = data_abide.load_without_size_preprocessing(input_folder = sys_config.orig_data_root_abide,
                                                                                    site_name = 'CALTECH',
                                                                                    idx = ids[sub_num],
                                                                                    depth = image_depth)
                num_rotations = 0

            elif test_dataset_name is 'STANFORD':
                # image will be normalized to [0,1]
                image_orig, labels_orig = data_abide.load_without_size_preprocessing(input_folder = sys_config.orig_data_root_abide,
                                                                                    site_name = 'STANFORD',
                                                                                    idx = ids[sub_num],
                                                                                    depth = image_depth)
                num_rotations = 0

            elif test_dataset_name is 'NCI':
                # image will be normalized to [0,1]
                image_orig, labels_orig = data_nci.load_without_size_preprocessing(sys_config.orig_data_root_nci,
                                                                                cv_fold_num=1,
                                                                                train_test='test',
                                                                                idx=ids[sub_num])
                num_rotations = 0

            elif test_dataset_name is 'PIRAD_ERC':
                # image will be normalized to [0,1]
                image_orig, labels_orig = data_pirad_erc.load_without_size_preprocessing(sys_config.orig_data_root_pirad_erc,
                                                                                        ids[sub_num],
                                                                                        labeller='ek')
                num_rotations = -3

            elif test_dataset_name is 'PROMISE':
                # image will be normalized to [0,1]
                image_orig, labels_orig = data_promise.load_without_size_preprocessing(sys_config.preproc_folder_promise,
                                                                                    subject_name[4:6])
                num_rotations = 0

            # ==================================================================
            # convert the predicitons back to original resolution
            # ==================================================================
            predicted_labels_orig_res_and_size = rescale_and_crop(predicted_labels,
                                                                orig_data_res_x[sub_num],
                                                                orig_data_res_y[sub_num],
                                                                orig_data_siz_x[sub_num],
                                                                orig_data_siz_y[sub_num],
                                                                order_interpolation = 0,
                                                                num_rotations = num_rotations)

            normalized_image_orig_res_and_size = rescale_and_crop(normalized_image,
                                                                orig_data_res_x[sub_num],
                                                                orig_data_res_y[sub_num],
                                                                orig_data_siz_x[sub_num],
                                                                orig_data_siz_y[sub_num],
                                                                order_interpolation = 1,
                                                                num_rotations = num_rotations)

            # ==================================================================
            # If only whole-gland comparisions are desired, merge the labels in both ground truth segmentations as well as the predictions
            # ==================================================================
            if exp_config.whole_gland_results is True:
                predicted_labels_orig_res_and_size[predicted_labels_orig_res_and_size!=0] = 1
                labels_orig[labels_orig!=0] = 1
                nl = 2
            else:
                nl = nlabels

            # ==================================================================
            # compute dice at the original resolution
            # ==================================================================    
            logging.info(labels_orig.shape)
            logging.info(predicted_labels_orig_res_and_size.shape)
            dice_per_label_this_subject = met.f1_score(labels_orig.flatten(),
                                                    predicted_labels_orig_res_and_size.flatten(),
                                                    average=None)
            
            # ==================================================================    
            # compute Hausforff distance at the original resolution
            # ==================================================================   
            compute_hsd = False
            if compute_hsd is True:
                hsd_per_label_this_subject = utils.compute_surface_distance(y1 = labels_orig,
                                                                            y2 = predicted_labels_orig_res_and_size,
                                                                            nlabels = nl)
            else:
                hsd_per_label_this_subject = np.zeros((nl))
            
            # ================================================================
            # save sample results
            # ================================================================
            d_vis = 32 # 256
            ids_vis = np.arange(0, 32, 4) # ids = np.arange(48, 256-48, (256-96)//8)
            if ds == True:
                image_orig = simulate_ds(image_orig, ds_type, ds_params, normalize_after_ds)
            utils_vis.save_sample_prediction_results(x = utils.crop_or_pad_volume_to_size_along_z(image_orig, d_vis),
                                                    x_norm = utils.crop_or_pad_volume_to_size_along_z(normalized_image_orig_res_and_size, d_vis),
                                                    y_pred = utils.crop_or_pad_volume_to_size_along_z(predicted_labels_orig_res_and_size, d_vis),
                                                    gt = utils.crop_or_pad_volume_to_size_along_z(labels_orig, d_vis),
                                                    num_rotations = - num_rotations, # rotate for consistent visualization across datasets
                                                    savepath = results_prefix + '_' + subject_name + '.png',
                                                    nlabels = nl,
                                                    ids=ids_vis)
                                    
            # ================================
            # write the mean fg dice of this subject to the text file
            # ================================
            results_file.write(subject_name + ":: dice (mean, std over all FG labels): ")
            results_file.write(str(np.round(np.mean(dice_per_label_this_subject[1:]), 3)) + ", " + str(np.round(np.std(dice_per_label_this_subject[1:]), 3)))
            results_file.write(", hausdorff distance (mean, std over all FG labels): ")
            results_file.write(str(np.round(np.mean(hsd_per_label_this_subject), 3)) + ", " + str(np.round(np.std(dice_per_label_this_subject[1:]), 3)) + "\n")
            
            dice_per_label_per_subject.append(dice_per_label_this_subject)
            hsd_per_label_per_subject.append(hsd_per_label_this_subject)
        
        # ================================================================
        # write per label statistics over all subjects    
        # ================================================================
        dice_per_label_per_subject = np.array(dice_per_label_per_subject)
        hsd_per_label_per_subject =  np.array(hsd_per_label_per_subject)
        
        # ================================
        # In the array images_dice, in the rows, there are subjects
        # and in the columns, there are the dice scores for each label for a particular subject
        # ================================
        results_file.write("================================== \n") 
        results_file.write("Label: dice mean, std. deviation over all subjects\n")
        for i in range(dice_per_label_per_subject.shape[1]):
            results_file.write(str(i) + ": " + str(np.round(np.mean(dice_per_label_per_subject[:,i]), 3)) + ", " + str(np.round(np.std(dice_per_label_per_subject[:,i]), 3)) + "\n")
        results_file.write("================================== \n") 
        results_file.write("Label: hausdorff distance mean, std. deviation over all subjects\n")
        for i in range(hsd_per_label_per_subject.shape[1]):
            results_file.write(str(i+1) + ": " + str(np.round(np.mean(hsd_per_label_per_subject[:,i]), 3)) + ", " + str(np.round(np.std(hsd_per_label_per_subject[:,i]), 3)) + "\n")
        
        # ==================
        # write the mean dice over all subjects and all labels
        # ==================
        results_file.write("================================== \n") 
        results_file.write("DICE Mean, std. deviation over foreground labels over all subjects: " + str(np.round(np.mean(dice_per_label_per_subject[:,1:]), 3)) + ", " + str(np.round(np.std(dice_per_label_per_subject[:,1:]), 3)) + "\n")
        results_file.write("HSD Mean, std. deviation over labels over all subjects: " + str(np.round(np.mean(hsd_per_label_per_subject), 3)) + ", " + str(np.round(np.std(hsd_per_label_per_subject), 3)) + "\n")
        results_file.write("================================== \n") 
        results_file.close()
        
# ==================================================================
# ==================================================================
if __name__ == '__main__':
    main()
