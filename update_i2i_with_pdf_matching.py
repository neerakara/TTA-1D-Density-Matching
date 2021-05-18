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

import argparse

# ==================================================================
# setup logging
# ==================================================================
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ==================================================================
# Set the config file of the experiment you want to run here:
# ==================================================================
from experiments import i2i as exp_config

# ===================================
# parse arguments
# =================================== 
parser = argparse.ArgumentParser(prog = 'PROG')

# read arguments
parser.add_argument('--test_sub_num', type = int, default = 0) # 0 to 19
parser.add_argument('--adaBN', type = int, default = 0) # 0 to 1
parser.add_argument('--tta_vars', default = "bn") # bn / norm
parser.add_argument('--match_moments', default = "first") # first / firsttwo / all
parser.add_argument('--b_size', type = int, default = 16) # 1 / 2 / 4 (requires 24G GPU)
parser.add_argument('--feature_subsampling_factor', type = int, default = 8) # 1 / 4
parser.add_argument('--features_randomized', type = int, default = 1) # 1 / 0
parser.add_argument('--batch_randomized', type = int, default = 1) # 1 / 0
parser.add_argument('--alpha', type = float, default = 100.0) # 100.0 / 1000.0
args = parser.parse_args()

target_resolution = exp_config.target_resolution
image_size = exp_config.image_size
nlabels = exp_config.nlabels

log_dir = os.path.join(sys_config.project_root, 'log_dir/' + exp_config.expname_i2l)
logging.info('SD training directory: %s' %log_dir)
tensorboard_dir = os.path.join(sys_config.tensorboard_root, exp_config.expname_i2l)

# ================================================================
# load training data
# ================================================================
if exp_config.train_dataset == 'NCI':
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

    orig_data_siz_z_train = data_pros['nz_train'][:]
    num_train_subjects = orig_data_siz_z_train.shape[0] 

# ================================================================
# load test data
# ================================================================
if exp_config.test_dataset == 'PROMISE':
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
# Run TTA for the asked subject
# ================================================================
for sub_num in range(args.test_sub_num, args.test_sub_num + 1):
    
    logging.info(str(name_test_subjects[sub_num])[2:-1])

    subject_name = str(name_test_subjects[sub_num])[2:-1]
    subject_string = exp_config.test_dataset + '_' + subject_name

    exp_str = exp_config.tta_string + 'tta_vars_' + args.tta_vars 
    exp_str = exp_str + '/moments_' + args.match_moments
    exp_str = exp_str + '_bsize' + str(args.b_size)
    exp_str = exp_str + '_rand' + str(args.batch_randomized)
    exp_str = exp_str + '_fs' + str(args.feature_subsampling_factor)
    exp_str = exp_str + '_rand' + str(args.features_randomized)
    exp_str = exp_str + '/' # _z_subsample
    exp_str = exp_str + subject_string
    log_dir_tta = log_dir + exp_str
    tensorboard_dir_tta = tensorboard_dir + exp_str
    
    logging.info('Tensorboard directory TTA: %s' %tensorboard_dir_tta)

    if not tf.gfile.Exists(log_dir_tta):
        tf.gfile.MakeDirs(log_dir_tta)
        tf.gfile.MakeDirs(tensorboard_dir_tta)
    
    subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])
    subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])

    test_image = imts[subject_id_start_slice:subject_id_end_slice,:,:]  
    test_image_gt = gtts[subject_id_start_slice:subject_id_end_slice,:,:]  

    # =============================================
    # These two subjects have an exceptionally higher resolution along the z-direction
    # All SD subjects have res along the z-direction = 3mm or 4mm
    # These two subjects have res along the z-direction ~ 2.2mm
    # =============================================
    # if (exp_config.test_dataset == 'PROMISE') and (sub_num in [8, 19]):
    #     # skipping every third slice for the TTA computations
    #     test_image = test_image[np.mod(np.arange(test_image.shape[0]), 3) != 0, :, :]
    #     test_image_gt = test_image_gt[np.mod(np.arange(test_image_gt.shape[0]), 3) != 0, :, :]

    test_image_gt = test_image_gt.astype(np.uint8)

    # ================================================================
    # build the TF graph
    # ================================================================
    with tf.Graph().as_default():
        
        # ================================================================
        # create placeholders
        # ================================================================
        images_pl = tf.placeholder(tf.float32, shape = [args.b_size] + list(image_size) + [1], name = 'images')
        training_pl = tf.constant(False, dtype=tf.bool)
        # ================================================================
        # insert a normalization module in front of the segmentation network
        # the normalization module is trained for each test image
        # ================================================================
        images_normalized, added_residual = model.normalize(images_pl, exp_config, training_pl = training_pl)
        # ================================================================
        # build the graph that computes predictions from the inference model
        # ================================================================
        logits, softmax, preds = model.predict_i2l(images_normalized, exp_config, training_pl = training_pl)
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

        if args.tta_vars == "bn":
            tta_vars = bn_vars
        elif args.tta_vars == "norm":
            tta_vars = normalization_vars

        # ================================================================
        # Define assignment ops for ADA-BN
        # ================================================================                 
        if args.adaBN == 1:               
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
        # Define BN stats matching loss
        # ================================================================
        # placeholder for SD PDFs (mean over all SD subjects). These will be extracted after loading the SD trained model.
        # The shapes have to be hard-coded. Can't get the tile operations to work otherwise..
        sd_pdf_pl = tf.placeholder(tf.float32, shape = [704, 61], name = 'sd_pdfs') # shape [num_channels, num_points_along_intensity_range]
        sd_pdf_std_pl = tf.placeholder(tf.float32, shape = [704, 61], name = 'sd_pdfs_std') # shape [num_channels, num_points_along_intensity_range]
        # placeholder for the points at which the PDFs are evaluated
        x_pdf_pl = tf.placeholder(tf.float32, shape = [61], name = 'x_pdfs') # shape [num_points_along_intensity_range]
        # placeholder for the smoothing factor in the KDE computation
        alpha_pl = tf.placeholder(tf.float32, shape = [], name = 'alpha') # shape [1]

        # ================================================================
        # compute the pdfs of features of the TD image that is fed via the placeholder
        # ================================================================
        td_pdfs = tf.zeros([1, sd_pdf_pl.shape[1]]) # shape [num_channels, num_points_along_intensity_range]
        for conv_block in [1,2,3,4,5,6,7]:
            for conv_sub_block in [1,2]:
                conv_string = str(conv_block) + '_' + str(conv_sub_block)
                features_td = tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '_bn/FusedBatchNorm:0')
                features_td = tf.reshape(features_td, (-1, features_td.shape[-1]))

                # for Batch size 2:
                # 1_1 (131072, 16), 1_2 (131072, 16), 2_1 (32768, 32), 2_2 (32768, 32)
                # 3_1 (8192, 64), 3_2 (8192, 64), 4_1 (2048, 128), 4_2 (2048, 128)
                # 5_1 (8192, 64), 5_2 (8192, 64), 6_1 (32768, 32), 6_2 (32768, 32)
                # 7_1 (131072, 16), 7_2 (131072, 16)

                # Subsample the feature maps to relieve the memory constraint and enable higher batch sizes
                if args.feature_subsampling_factor != 1:
                    if args.features_randomized == 0:
                        features_td = features_td[::args.feature_subsampling_factor, :]
                    elif args.features_randomized == 1:
                        # https://stackoverflow.com/questions/49734747/how-would-i-randomly-sample-pixels-in-tensorflow
                        # https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/gather.md
                        random_indices = tf.random.uniform(shape=[features_td.shape[0].value // args.feature_subsampling_factor],
                                                           minval=0,
                                                           maxval=features_td.shape[0].value - 1,
                                                           dtype=tf.int32)
                        features_td = tf.gather(features_td, random_indices, axis=0)

                features_td = tf.tile(tf.expand_dims(features_td, 0), multiples = [x_pdf_pl.shape[0], 1, 1])
                x_pdf_tmp = tf.tile(tf.expand_dims(tf.expand_dims(x_pdf_pl, -1), -1), multiples = [1, features_td.shape[1], features_td.shape[2]])

                # the 3 dimensions are : 
                # 1. the intensity values where the pdf is evaluated,
                # 2. all the features (the pixels along the 2 spatial dimensions as well as the batch dimension are considered 1D iid samples)
                # 3. the channels 
                channel_pdf_this_layer_td = tf.reduce_mean(tf.math.exp(-alpha_pl * tf.math.square(x_pdf_tmp - features_td)), axis=1)
                channel_pdf_this_layer_td = tf.transpose(channel_pdf_this_layer_td)
                # at the end, we get 1 pdf (evaluated at the intensity values in x_pdf_pl) per channel
                
                td_pdfs = tf.concat([td_pdfs, channel_pdf_this_layer_td], 0)
        
        # ignore the zeroth column that was added at the start of the loop
        td_pdfs = td_pdfs[1:, :]

        # ================================================================
        # compute the TTA loss - add ops for all losses and select based on the argument
        # ================================================================

        # L2 distance between PDFs
        loss_all_op = tf.reduce_mean(tf.math.square(td_pdfs - sd_pdf_pl)) # mean over all channels of all layers

        # L2 distance between PDFs, with each coordinate scaled according to the log-variance across the SD subjects at that intensity value.
        epsilon = 1e-10
        loss_all_std_op = tf.reduce_mean(tf.math.square(tf.math.divide(td_pdfs - sd_pdf_pl, tf.math.log(sd_pdf_std_pl + epsilon)))) # mean over all channels of all layers

        # compute means from the PDFs : $ \mu = \sum_{i=xmin}^{xmax} x * p(x) $
        x_pdf_tiled = tf.tile(tf.expand_dims(x_pdf_pl, 0), multiples = [td_pdfs.shape[0], 1]) # [Nc, Nx]
        td_pdf_means = tf.reduce_sum(tf.math.multiply(td_pdfs, x_pdf_tiled), axis = 1) # [Nc]
        sd_pdf_means = tf.reduce_sum(tf.math.multiply(sd_pdf_pl, x_pdf_tiled), axis = 1) # [Nc]
        loss_one_op = tf.reduce_mean(tf.math.square(td_pdf_means - sd_pdf_means)) # [Nc] (before reduce_mean)

        # compute variances from the PDFs, using the means computed above
        # $ \sigma^2 = \sum_{i=xmin}^{xmax} (x - \mu)^2 * p(x) $
        td_pdf_variances_tmp = tf.math.square(x_pdf_tiled - tf.tile(tf.expand_dims(td_pdf_means, 1), multiples = [1, x_pdf_tiled.shape[1]]))
        td_pdf_variances = tf.reduce_sum(tf.math.multiply(td_pdfs, td_pdf_variances_tmp), axis = 1) # [Nc]
        sd_pdf_variances_tmp = tf.math.square(x_pdf_tiled - tf.tile(tf.expand_dims(sd_pdf_means, 1), multiples = [1, x_pdf_tiled.shape[1]]))
        sd_pdf_variances = tf.reduce_sum(tf.math.multiply(sd_pdf_pl, sd_pdf_variances_tmp), axis = 1) # [Nc]
        loss_onetwo_op = tf.reduce_mean(tf.math.square(td_pdf_means - sd_pdf_means) + tf.math.square(td_pdf_variances - sd_pdf_variances))
        loss_onetwokl_op = tf.reduce_mean(tf.math.log(td_pdf_variances / sd_pdf_variances) + (sd_pdf_variances + (sd_pdf_means - td_pdf_means)**2) / td_pdf_variances)

        # compute CFs of the source and target domains
        td_cfs = tf.spectral.rfft(td_pdfs)
        sd_cfs = tf.spectral.rfft(sd_pdf_pl)

        # min L2 distance between complex arrays (match CFs exactly)
        # TODO: Check how L2 distance is defined from complex arrays
        loss_all_cf_real_op = tf.reduce_mean(tf.math.square(tf.math.real(td_cfs) - tf.math.real(sd_cfs))) # mean over all channels of all layers
        loss_all_cf_imag_op = tf.reduce_mean(tf.math.square(tf.math.imag(td_cfs) - tf.math.imag(sd_cfs))) # mean over all channels of all layers
        loss_all_cf_op = loss_all_cf_real_op + loss_all_cf_imag_op

        # min L2 distance between magnitudes of complex arrays (match only which frequencies are contained in the CFs, phase can be different.)
        # IDEA: If the modes of the PDF are a bit shifted - this is fine, but if the SD consists of 2 modes, the TD should also have 2 modes corresponding to the same frequecies.
        loss_all_cf_mag_only_op = tf.reduce_mean(tf.math.square(tf.math.abs(td_cfs) - tf.math.abs(sd_cfs))) # mean over all channels of all layers
        
        # match the PDFs 
        if args.match_moments == 'all': 
            loss_op = loss_all_op
        # match the PDFs, with less weight for points where the variance over the SD subject is high
        elif args.match_moments == 'all_std_log': 
            loss_op = loss_all_std_op
        # match the means of the PDFs
        elif args.match_moments == 'first': 
            loss_op = loss_one_op    
        # match the means and standard deviations of the PDFs
        elif args.match_moments == 'firsttwo': 
            loss_op = loss_onetwo_op
        # match the means and standard deviations of the PDFs, by minimizing the kl div between the 1d gaussians
        elif args.match_moments == 'firsttwo_kl':
            loss_op = loss_onetwokl_op
        # min L2 distance between complex arrays (match CFs exactly)
        elif args.match_moments == 'CF':
            loss_op = loss_all_cf_op
        # min L2 distance between magnitudes of complex arrays (match only which frequencies are contained in the CFs, phase can be different.)
        elif args.match_moments == 'CF_mag':
            loss_op = loss_all_cf_mag_only_op
                
        # ================================================================
        # add losses to tensorboard
        # ================================================================
        tf.summary.scalar('loss/tta', loss_op)         
        tf.summary.scalar('loss/1D_all', loss_all_op)
        tf.summary.scalar('loss/1D_all_std', loss_all_std_op)
        tf.summary.scalar('loss/1D_one', loss_one_op)
        tf.summary.scalar('loss/1D_onetwo', loss_onetwo_op)
        tf.summary.scalar('loss/1D_onetwokl', loss_onetwokl_op)
        tf.summary.scalar('loss/1D_all_cf', loss_all_cf_op)
        tf.summary.scalar('loss/1D_all_cf_mag', loss_all_cf_mag_only_op)
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
        display_pdfs_pl = tf.placeholder(tf.uint8, shape = [1, None, None, 1], name='display_pdfs_pl')
        pdfs_summary = tf.summary.image('PDFs', display_pdfs_pl)
        display_cfs_abs_pl = tf.placeholder(tf.uint8, shape = [1, None, None, 1], name='display_cfs_abs_pl')
        cfs_abs_summary = tf.summary.image('CFs_Magnitude', display_cfs_abs_pl)
        display_cfs_angle_pl = tf.placeholder(tf.uint8, shape = [1, None, None, 1], name='display_cfs_angle_pl')
        cfs_angle_summary = tf.summary.image('CFs_Phase', display_cfs_angle_pl)

        # ================================================================
        # create saver
        # ================================================================
        saver_i2l = tf.train.Saver(var_list = i2l_vars)
        saver_tta = tf.train.Saver(var_list = tta_vars, max_to_keep=3)   
                
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
        # compute the SD PDFs once (extract the whole pdf instead of just the 1st and 2nd moments of the pdf), and pass them as placeholders for computing the loss in each iteration
        # ================================================================
        b_size = args.b_size
        alpha = args.alpha
        res = 0.1
        x_min = -3.0
        x_max = 3.0
        pdf_str = 'alpha' + str(alpha) + 'xmin' + str(x_min) + 'xmax' + str(x_max) + '_res' + str(res) + '_bsize2' # + str(b_size)
        x_values = np.arange(x_min, x_max + res, res)
        
        sd_pdfs_filename = path_to_model + 'sd_pdfs_' + pdf_str + '_mean_and_variance.npy'
        
        if os.path.isfile(sd_pdfs_filename):            
            pdfs_sd = np.load(sd_pdfs_filename) # [num_subjects, num_channels, num_x_points]
        
        else:
            pdfs_sd = []
            num_training_subjects = orig_data_siz_z_train.shape[0]            
            for train_sub_num in range(num_training_subjects):            
                
                logging.info("==== Computing pdf for subject " + str(train_sub_num) + '..')
                sd_image = imtr[np.sum(orig_data_siz_z_train[:train_sub_num]) : np.sum(orig_data_siz_z_train[:train_sub_num+1]),:,:]
                logging.info(sd_image.shape)
                
                num_batches = 0
                
                for b_i in range(0, sd_image.shape[0], b_size):
                    if b_i + b_size < sd_image.shape[0]: # ignoring the rest of the image (that doesn't fit the last batch) for now.
                        pdfs_this_batch = sess.run(td_pdfs, feed_dict={images_pl: np.expand_dims(sd_image[b_i:b_i+b_size, ...], axis=-1),
                                                                       x_pdf_pl: x_values,
                                                                       alpha_pl: alpha})
                        if b_i == 0:
                            pdfs_this_subject = pdfs_this_batch

                        else:
                            pdfs_this_subject = pdfs_this_subject + pdfs_this_batch

                        num_batches = num_batches + 1
                
                pdfs_this_subject = pdfs_this_subject / num_batches

                pdfs_sd.append(pdfs_this_subject)

            #     # ================================================================      
            #     # Keep summing the PDFs of individual subjects
            #     # ================================================================      
            #     if train_sub_num == 0:
            #         pdfs_sd = pdfs_this_subject
            #     else:
            #         pdfs_sd = pdfs_sd + pdfs_this_subject

            # # ================================================================
            # # Assuming same number of pixels in all subjects, the SD pdf will be the average of all pdfs
            # # ================================================================
            # pdfs_sd = pdfs_sd / num_training_subjects

            pdfs_sd = np.array(pdfs_sd)

            # ================================================================
            # save
            # ================================================================
            np.save(sd_pdfs_filename, pdfs_sd) # [num_subjects, num_channels, num_x_points]

        pdfs_sd_mean = np.mean(pdfs_sd, axis = 0)
        pdfs_sd_std = np.std(pdfs_sd, axis = 0)
        logging.info(np.max(pdfs_sd_mean))
        logging.info(np.max(pdfs_sd_std))
        # variance is quite small compared to the mean at each point on the PDF
        # --> This is true even when data augmentation is used while computing the SD PDFs
        # --> Seems like the network maps all the training images to similar features already from the first layers... (!?)

        # ================================================================
        # ADA-BN
        # ================================================================
        # Compute mean and variance of the current 3D test image for features of each channel in each layer.
        # Replace the SD mean and SD variance stored in the BN layers with the computed values.
        # Do this one layer at a time - that is first replace the means and variances in the first BN layer.
        # Now, compute the means and variances of the TD features at the second layer and replace them.
        # Next, follow the same step for the next layers one by one.
        # ================================================================
        if args.adaBN == 1:

            count = 0
            for conv_block in [1,2,3,4,5,6,7]:
                for conv_sub_block in [1,2]:
                    # divide the image into batches
                    b_size = args.b_size
                    num_batches = 0
                    for b_i in range(0, test_image.shape[0], b_size):
                        if b_i + b_size < test_image.shape[0]: # ignoring the rest of the image (that doesn't fit the last batch) for now.
                            batch = np.expand_dims(test_image[b_i:b_i+b_size, ...], axis=-1)
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
                    # Assign these values to moving means and moving variances before doing the TTA
                    # This will match the first two moments, and then the TTA will try to match the higher order moments
                    # =============
                    sess.run(mean_assign_ops[count], feed_dict={tmp_mean_pl: td_means_this_layer})
                    sess.run(variance_assign_ops[count], feed_dict={tmp_variance_pl: td_variances_this_layer})                
                    count = count + 1
                    
        # ================================================================
        # TTA iterations
        # ================================================================
        step = 0
        best_loss = 1000.0
        while (step < exp_config.max_steps_i2i):
            
            logging.info("TTA step: " + str(step+1))
            
            # run accumulated_gradients_zero_op (no need to provide values for any placeholders)
            sess.run(accumulated_gradients_zero_op)
            num_accumulation_steps = 0
            loss_this_step = 0.0
            
            b_size = args.b_size
            for b_i in range(0, test_image.shape[0], b_size):

                if args.batch_randomized == 0:
                    if b_i + b_size < test_image.shape[0]: # ignoring the rest of the image (that doesn't fit the last batch) for now.
                        # run the accumulate gradients op 
                        feed_dict={images_pl: np.expand_dims(test_image[b_i:b_i+b_size, :, :], axis=-1),
                                sd_pdf_pl: pdfs_sd_mean, 
                                sd_pdf_std_pl: pdfs_sd_std,
                                x_pdf_pl: x_values, 
                                alpha_pl: alpha}
                        sess.run(accumulate_gradients_op, feed_dict=feed_dict)
                        loss_this_step = loss_this_step + sess.run(loss_op, feed_dict = feed_dict)
                        num_accumulation_steps = num_accumulation_steps + 1

                elif args.batch_randomized == 1:      
                        # run the accumulate gradients op 
                        feed_dict={images_pl: np.expand_dims(test_image[np.random.randint(0, test_image.shape[0], b_size), :, :], axis=-1),
                                   sd_pdf_pl: pdfs_sd_mean, 
                                   sd_pdf_std_pl: pdfs_sd_std,
                                   x_pdf_pl: x_values, 
                                   alpha_pl: alpha}
                        sess.run(accumulate_gradients_op, feed_dict=feed_dict)
                        loss_this_step = loss_this_step + sess.run(loss_op, feed_dict = feed_dict)
                        num_accumulation_steps = num_accumulation_steps + 1

            # ===========================
            # save best model so far
            # ===========================
            loss_this_step = loss_this_step / num_accumulation_steps # average loss (over all slices of the image volume) in this step
            if best_loss > loss_this_step:
                best_loss = loss_this_step
                best_file = os.path.join(log_dir_tta, 'models/best_loss.ckpt')
                saver_tta.save(sess, best_file, global_step=step)
                logging.info('Found new best score (%f) at step %d -  Saving model.' % (best_loss, step))

            # ===========================
            # run accumulated_gradients_mean_op, with a value for the placeholder num_accumulation_steps_pl; followed by the train_op with applies the gradients
            # ===========================
            sess.run(accumulated_gradients_mean_op, feed_dict = {num_accumulation_steps_pl: num_accumulation_steps})
            # run the train_op. this also requires input output placeholders, as compute_gradients will be called again, but the returned gradient values will be replaced by the mean gradients.
            sess.run(train_op, feed_dict = feed_dict)

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

            if b_size > 1:
                label_predicted = np.reshape(label_predicted, (label_predicted.shape[0]*label_predicted.shape[1], label_predicted.shape[2], label_predicted.shape[3]))
                image_normalized = np.reshape(image_normalized, (image_normalized.shape[0]*image_normalized.shape[1], image_normalized.shape[2], image_normalized.shape[3]))
                label_predicted = label_predicted[:test_image.shape[0], ...]
                image_normalized = image_normalized[:test_image.shape[0], ...]

            if exp_config.test_dataset is 'PROMISE':
                label_predicted[label_predicted!=0] = 1
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
            if step % 10 == 0:

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
                b_size = args.b_size
                num_batches = 0
                for b_i in range(0, test_image.shape[0], b_size):
                    if b_i + b_size < test_image.shape[0]: # ignoring the rest of the image (that doesn't fit the last batch) for now.
                        pdfs_this_batch = sess.run(td_pdfs, feed_dict={images_pl: np.expand_dims(test_image[b_i:b_i+b_size, ...], axis=-1),
                                                                       x_pdf_pl: x_values,
                                                                       alpha_pl: alpha})
                        if b_i == 0:
                            pdfs_td_this_step = pdfs_this_batch
                        else:
                            pdfs_td_this_step = pdfs_td_this_step + pdfs_this_batch
                        num_batches = num_batches + 1
                pdfs_td_this_step = pdfs_td_this_step / num_batches

                utils_vis.write_pdfs(step,
                                     summary_writer,
                                     sess,
                                     pdfs_summary,
                                     display_pdfs_pl,
                                     pdfs_sd_mean,
                                     pdfs_sd_std,
                                     pdfs_td_this_step,
                                     x_values,
                                     log_dir_tta)

                # ===========================
                # visualize feature distribution alignment
                # ===========================
                b_i = 0
                sd_cfs_batch_this_step = sess.run(sd_cfs, feed_dict={sd_pdf_pl: pdfs_sd_mean})
                td_cfs_batch_this_step = sess.run(td_cfs, feed_dict={images_pl: np.expand_dims(test_image[b_i:b_i+b_size, ...], axis=-1),
                                                                     x_pdf_pl: x_values,
                                                                     alpha_pl: alpha})

                logging.info(sd_cfs_batch_this_step.shape)
                logging.info(sd_cfs_batch_this_step.dtype)
                logging.info(td_cfs_batch_this_step.shape)
                logging.info(td_cfs_batch_this_step.dtype)

                utils_vis.write_cfs(step,
                                    summary_writer,
                                    sess,
                                    cfs_abs_summary,
                                    cfs_angle_summary,
                                    display_cfs_abs_pl,
                                    display_cfs_angle_pl,
                                    sd_cfs_batch_this_step,
                                    td_cfs_batch_this_step,
                                    log_dir_tta)

            step = step + 1

        # ================================================================
        # close session
        # ================================================================
        sess.close()



# ================================================================
# ================================================================
# PDF computation in numpy.. was taking 5 minutes per 3D volume.
# ================================================================
# ================================================================
# compute_channel_wise = True

# if compute_channel_wise == True:
#     # ================================================================      
#     # For loop over the channels in each layer
#     # ================================================================          
#     channel_pdfs = []
#     for conv_block in [1,2,3,4,5,6,7]:
#         for conv_sub_block in [1, 2]:
#             conv_string = str(conv_block) + '_' + str(conv_sub_block)
#             # get features after the BN layer. (due to this, the features are roughly around zero and with a similar standard deviation for all layers)
#             features_sd = []
#             for b_i in range(0, sd_image.shape[0], 1):
#                 features_sd.append(sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '_bn/FusedBatchNorm:0'),
#                                    feed_dict={images_pl: np.expand_dims(sd_image[b_i:b_i+1, ...], axis=-1)}))
#             features_sd = np.squeeze(np.array(features_sd)).astype(float)  
#             features_sd = np.reshape(features_sd, (-1, features_sd.shape[-1]))
#             logging.info("==== Feature" + conv_string + " extracted from a source domain image:" + str(features_sd.shape) + ", range: " + str(np.min(features_sd)) + ", " + str(np.max(features_sd)))

#             for channel in range(features_sd.shape[-1]):
#                 features_sd_this_channel = np.tile(features_sd[:,channel], (x_values.shape[0], 1))
#                 x_tmp = np.tile(x_values, (features_sd.shape[0], 1)).T
#                 pdf_this_channel = np.sum(np.exp(-alpha * (x_tmp - features_sd_this_channel) ** 2), -1) / features_sd.shape[0]
#                 channel_pdfs.append(pdf_this_channel)
#                 # visualize some channels 
#                 if channel % 16 == 0:
#                     utils_vis.plot_graph(x_values, pdf_this_channel, log_dir + exp_config.tta_string + 'pdf_sub' + str(train_sub_num) + '_' + conv_string + '_channel' + str(channel) + '_alpha' + str(alpha) + '.png')
#     channel_pdfs = np.array(channel_pdfs)

# else:
#     # ================================================================      
#     # Computing pdfs for all channels of a layer simultaneously
#     # ================================================================      
#     channel_pdfs = np.zeros((x_values.shape[0], 1))
#     for conv_block in [1,2,3,4,5,6,7]:
#         for conv_sub_block in [1, 2]:
#             conv_string = str(conv_block) + '_' + str(conv_sub_block)
#             # get features after the BN layer. (due to this, the features are roughly around zero and with a similar standard deviation for all layers)
#             features_sd = []
#             for b_i in range(0, sd_image.shape[0], 1):
#                 features_sd.append(sess.run(tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '_bn/FusedBatchNorm:0'),
#                                    feed_dict={images_pl: np.expand_dims(sd_image[b_i:b_i+1, ...], axis=-1)}))
#             features_sd = np.squeeze(np.array(features_sd)).astype(float)  
#             features_sd = np.reshape(features_sd, (-1, features_sd.shape[-1]))
#             logging.info("==== Feature" + conv_string + " extracted from a source domain image:" + str(features_sd.shape) + ", range: " + str(np.min(features_sd)) + ", " + str(np.max(features_sd)))
#             features_sd_this_layer = np.tile(features_sd, (x_values.shape[0], 1, 1))
#             x_tmp = np.swapaxes(np.swapaxes(np.tile(x_values, (features_sd.shape[0], features_sd.shape[1], 1)), 1, 2), 0, 1)
#             channel_pdf_this_layer = np.sum(np.exp(-alpha * (x_tmp - features_sd_this_layer) ** 2), 1) / features_sd.shape[0]
#             channel_pdfs = np.concatenate((channel_pdfs, channel_pdf_this_layer), -1)
#             # visualize some channels 
#             for channel in range(channel_pdf_this_layer.shape[-1] // 16):
#                 utils_vis.plot_graph(x_values, channel_pdf_this_layer[:, channel], log_dir + exp_config.tta_string + 'pdf_sub' + str(train_sub_num) + '_' + conv_string + '_channel' + str(channel) + '_alpha' + str(alpha) + '.png')
#     channel_pdfs = np.delete(channel_pdfs, 0, axis = -1).T