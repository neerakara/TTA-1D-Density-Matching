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
import config.system_paths as sys_config
import config.params as exp_config
from skimage.transform import rescale
import sklearn.metrics as met
import argparse

# ==================================================================
# parse arguments
# ==================================================================
parser = argparse.ArgumentParser(prog = 'PROG')
# Training dataset and run number
parser.add_argument('--train_dataset', default = "NCI") # NCI / HCPT1
parser.add_argument('--tr_run_number', type = int, default = 1) # 1 / 
# Test dataset 
parser.add_argument('--test_dataset', default = "USZ") # PROMISE / USZ / CALTECH / STANFORD / HCPT2
parser.add_argument('--NORMALIZE', type = int, default = 1) # 1 / 0
# TTA options
parser.add_argument('--tta_string', default = "TTA/")
parser.add_argument('--adaBN', type = int, default = 0) # 0 to 1
# Whether to compute KDE or not?
parser.add_argument('--KDE', type = int, default = 1) # 0 to 1
parser.add_argument('--alpha', type = float, default = 100.0) # 10.0 / 100.0 / 1000.0
parser.add_argument('--KDE_Groups', type = int, default = 1) # 0 / 1
# Which vars to adapt?
parser.add_argument('--tta_vars', default = "NORM") # BN / NORM
# How many moments to match and how?
parser.add_argument('--match_moments', default = "All_KL") # Gaussian_KL / All_KL / All_CF_L2
parser.add_argument('--before_or_after_bn', default = "AFTER") # AFTER / BEFORE
# Batch settings
parser.add_argument('--b_size', type = int, default = 16) # 1 / 2 / 4 (requires 24G GPU)
parser.add_argument('--feature_subsampling_factor', type = int, default = 16) # 1 / 4
parser.add_argument('--features_randomized', type = int, default = 1) # 1 / 0
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
whole_gland_results = dataset_params[5]

# ================================================================
# Make the name for this TTA run
# ================================================================
exp_str = exp_config.make_tta_exp_name(args)

# ================================================================
# Setup directories for this run
# ================================================================
expname_i2l = 'tr' + args.train_dataset + '_r' + str(args.tr_run_number) + '/' + 'i2i2l/'
log_dir_sd = sys_config.project_root + 'log_dir/' + expname_i2l
log_dir_tta = log_dir_sd + exp_str

# ==================================================================
# Identifier for SFDA
# ==================================================================
if args.TTA_or_SFDA == 'SFDA':
    if args.test_dataset == 'USZ':
        td_string = 'SFDA_' + args.test_dataset + '/'
    elif args.test_dataset == 'PROMISE':    
        td_string = 'SFDA_' + args.test_dataset + '_' + args.PROMISE_SUB_DATASET + '/'

# ==================================================================
# main function for inference
# ==================================================================
def predict_segmentation(subject_name,
                         image,
                         normalize = 1):
    
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
                                                   training_pl = tf.constant(False, dtype=tf.bool),
                                                   nlabels = nlabels)
                        
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

        if args.tta_vars == 'BN':
            tta_vars = bn_vars
        elif args.tta_vars == 'NORM':
            tta_vars = normalization_vars
                                
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
        saver_tta = tf.train.Saver(var_list = tta_vars)        
                
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
        if normalize == 1:
            logging.info('============================================================')
            if args.TTA_or_SFDA == 'TTA':
                subject_string = args.test_dataset + '_' + subject_name + '/'
                path_to_model = log_dir_tta + subject_string + 'models/'
            elif args.TTA_or_SFDA == 'SFDA':
                path_to_model = log_dir_tta + td_string + 'models/'

            checkpoint_path = utils.get_latest_model_checkpoint_path(path_to_model, 'best_loss.ckpt')
            logging.info('Restoring the trained parameters from %s...' % checkpoint_path)
            saver_tta.restore(sess, checkpoint_path)
            logging.info('============================================================')

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
                                 mode = 'constant',
                                 anti_aliasing = False)

        arr2d_rescaled_cropped = utils.crop_or_pad_slice_to_size(arr2d_rescaled, nx, ny)

        arr_list.append(arr2d_rescaled_cropped)
    
    arr_orig_res_and_size = np.array(arr_list)
    arr_orig_res_and_size = arr_orig_res_and_size.swapaxes(0, 1).swapaxes(1, 2)
    
    return arr_orig_res_and_size
        
# ==================================================================
# ==================================================================
def main():
    
    # ===================================
    # read the test images
    # ===================================
    test_dataset_name = args.test_dataset
    loaded_test_data = utils.load_testing_data(test_dataset_name,
                                               image_size,
                                               target_resolution,
                                               image_depth_ts)

    imts = loaded_test_data[0]
    orig_data_res_x = loaded_test_data[2]
    orig_data_res_y = loaded_test_data[3]
    orig_data_res_z = loaded_test_data[4]
    orig_data_siz_x = loaded_test_data[5]
    orig_data_siz_y = loaded_test_data[6]
    orig_data_siz_z = loaded_test_data[7]
    name_test_subjects = loaded_test_data[8]
    num_test_subjects = loaded_test_data[9]
    ids = loaded_test_data[10]
            
    # ================================   
    # open a text file for writing the mean dice scores for each subject that is evaluated
    # ================================
    if args.NORMALIZE == 1:
        if args.TTA_or_SFDA == 'TTA':
            results_filename = log_dir_tta + test_dataset_name + '_test'
        elif args.TTA_or_SFDA == 'SFDA':
            results_filename = log_dir_tta + td_string + test_dataset_name + '_test'
    else:
        results_filename = log_dir_sd + test_dataset_name + '_test'
    
    if whole_gland_results == True:
        results_filename = results_filename + '_whole_gland'
    
    results_file = open(results_filename + '.txt', "w")
    results_file.write("================================== \n") 
    results_file.write("Test results \n") 
    
    # ================================================================
    # For each test image, load the best model and compute the dice with this model
    # ================================================================
    dice_per_label_per_subject = []
    hsd_per_label_per_subject = []

    for sub_num in range(num_test_subjects):

        subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])
        subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])
        image = imts[subject_id_start_slice:subject_id_end_slice,:,:]  
        
        # ==================================================================
        # setup logging
        # ==================================================================
        logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')
        subject_name = str(name_test_subjects[sub_num])[2:-1]
        logging.info('============================================================')
        logging.info('Subject id: %s' %sub_num)
        logging.info(subject_name)

        # If the 'models' directory does not exist for this subject, move onto the next one
        if args.NORMALIZE == 1:
            if not tf.gfile.Exists(log_dir_tta + test_dataset_name + '_' + subject_name + '/models/'):
                continue
    
        # ==================================================================
        # predict segmentation at the pre-processed resolution
        # ==================================================================
        predicted_labels, normalized_image = predict_segmentation(subject_name,
                                                                  image,
                                                                  args.NORMALIZE)
                
        # ==================================================================
        # read the original segmentation mask
        # ==================================================================
        image_orig, labels_orig, num_rotations = utils.load_testing_data_wo_preproc(test_dataset_name,
                                                                                    ids,
                                                                                    sub_num,
                                                                                    subject_name,
                                                                                    image_depth_ts)

        # ==================================================================
        # convert the predicitons back to original resolution
        # ==================================================================
        predicted_labels_orig_res_and_size = rescale_and_crop(predicted_labels,
                                                              orig_data_res_x[sub_num],
                                                              orig_data_res_y[sub_num],
                                                              orig_data_siz_x[sub_num],
                                                              orig_data_siz_y[sub_num],
                                                              order_interpolation = 0,
                                                              num_rotations = num_rotations).astype(np.uint8)

        # ==================================================================
        # Name of visualization
        # ==================================================================
        if args.NORMALIZE == 1:
            if args.TTA_or_SFDA == 'TTA':
                savepath = log_dir_tta + test_dataset_name + '_test_' + subject_name
            elif args.TTA_or_SFDA == 'SFDA':
                savepath = log_dir_tta + td_string + test_dataset_name + '_test_' + subject_name
        else:
            savepath = log_dir_sd + test_dataset_name + '_test_' + subject_name

        # ==================================================================
        # If only whole-gland comparisions are desired, merge the labels in both ground truth segmentations as well as the predictions
        # ==================================================================
        if whole_gland_results == True:
            predicted_labels_orig_res_and_size[predicted_labels_orig_res_and_size!=0] = 1
            labels_orig[labels_orig!=0] = 1
            nl = 2
            savepath = savepath + '_whole_gland'
        else:
            nl = nlabels

        # ==================================================================
        # compute dice at the original resolution
        # ==================================================================    
        logging.info(labels_orig.shape)
        logging.info(predicted_labels_orig_res_and_size.shape)
        logging.info(labels_orig.dtype)
        logging.info(predicted_labels_orig_res_and_size.dtype)
        dice_per_label_this_subject = met.f1_score(labels_orig.flatten(),
                                                   predicted_labels_orig_res_and_size.flatten(),
                                                   average=None)
        
        # ==================================================================    
        # compute Hausforff distance at the original resolution
        # ==================================================================   
        compute_hsd = False
        if compute_hsd == True:
            hsd_per_label_this_subject = utils.compute_surface_distance(y1 = labels_orig,
                                                                        y2 = predicted_labels_orig_res_and_size,
                                                                        nlabels = nl)
        else:
            hsd_per_label_this_subject = np.zeros((nl))
        
        # ================================================================
        # save sample results
        # ================================================================
        save_visual_results = False
        if save_visual_results == True:
            d_vis = 32 # 256
            ids_vis = np.arange(0, 32, 4) # ids = np.arange(48, 256-48, (256-96)//8)
            utils_vis.save_sample_prediction_results(x = utils.crop_or_pad_volume_to_size_along_z(image_orig, d_vis),
                                                    x_norm = utils.crop_or_pad_volume_to_size_along_z(image_orig, d_vis),
                                                    y_pred = utils.crop_or_pad_volume_to_size_along_z(predicted_labels_orig_res_and_size, d_vis),
                                                    gt = utils.crop_or_pad_volume_to_size_along_z(labels_orig, d_vis),
                                                    num_rotations = - num_rotations, # rotate for consistent visualization across datasets
                                                    savepath = savepath + '.png',
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
