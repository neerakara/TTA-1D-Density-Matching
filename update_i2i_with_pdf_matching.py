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
parser.add_argument('--train_dataset', default = "NCI") # NCI
parser.add_argument('--test_dataset', default = "USZ") # PROMISE / USZ
parser.add_argument('--test_sub_num', type = int, default = 0) # 0 to 19
parser.add_argument('--adaBN', type = int, default = 0) # 0 to 1
parser.add_argument('--tta_vars', default = "norm") # bn / norm
parser.add_argument('--match_moments', default = "all_kl") # first / firsttwo / all
parser.add_argument('--b_size', type = int, default = 16) # 1 / 2 / 4 (requires 24G GPU)
parser.add_argument('--feature_subsampling_factor', type = int, default = 8) # 1 / 4
parser.add_argument('--features_randomized', type = int, default = 1) # 1 / 0
parser.add_argument('--batch_randomized', type = int, default = 1) # 1 / 0
parser.add_argument('--match_with_sd', type = int, default = 2) # 1 / 2 / 3 / 4
parser.add_argument('--alpha', type = float, default = 100.0) # 100.0 / 1000.0
parser.add_argument('--TTA_or_SFDA', default = "SFDA") # TTA / SFDA
parser.add_argument('--PROMISE_SUB_DATASET', default = "RUNMC") # RUNMC / UCL / BIDMC / HK
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
if args.train_dataset == 'NCI':
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
# load PROMISE
# ================================================================
if args.test_dataset == 'PROMISE':
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

# ================================================================
# load USZ
# ================================================================
elif args.test_dataset == 'USZ':

    image_depth = 32
    z_resolution = 2.5

    data_pros = data_pirad_erc.load_data(input_folder = sys_config.orig_data_root_pirad_erc,
                                         preproc_folder = sys_config.preproc_folder_pirad_erc,
                                         idx_start = 0,
                                         idx_end = 20,
                                         size = image_size,
                                         target_resolution = target_resolution,
                                         labeller = 'ek')
    
    imts, gtts = [data_pros['images'], data_pros['labels']]
    name_test_subjects = data_pros['patnames']
    orig_data_siz_z = data_pros['nz'][:]
    orig_data_res_z = data_pros['pz'][:]
    num_test_subjects = orig_data_siz_z.shape[0] 

# ================================================================
# Run TTA for the asked subject / SFDA for the requested TD
# ================================================================
exp_str = exp_config.tta_string + 'tta_vars_' + args.tta_vars 
exp_str = exp_str + '/moments_' + args.match_moments
exp_str = exp_str + '_bsize' + str(args.b_size)
exp_str = exp_str + '_rand' + str(args.batch_randomized)
exp_str = exp_str + '_fs' + str(args.feature_subsampling_factor)
exp_str = exp_str + '_rand' + str(args.features_randomized)
exp_str = exp_str + '_sd_match' + str(args.match_with_sd)
exp_str = exp_str + '/' # _z_subsample

if args.TTA_or_SFDA == 'TTA':
    sub_num = args.test_sub_num    
    logging.info(str(name_test_subjects[sub_num])[2:-1])
    subject_name = str(name_test_subjects[sub_num])[2:-1]
    subject_string = args.test_dataset + '_' + subject_name
    exp_str = exp_str + subject_string

    # extract the single test volume
    subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])
    subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])
    test_image = imts[subject_id_start_slice:subject_id_end_slice,:,:]  
    test_image_gt = gtts[subject_id_start_slice:subject_id_end_slice,:,:]  
    test_image_gt = test_image_gt.astype(np.uint8)

elif args.TTA_or_SFDA == 'SFDA':
    if args.test_dataset == 'USZ':
        td_string = 'SFDA_' + args.test_dataset
        test_ids = np.arange(num_test_subjects)

    elif args.test_dataset == 'PROMISE':
        td_string = 'SFDA_' + args.test_dataset + '_' + args.PROMISE_SUB_DATASET
        if args.PROMISE_SUB_DATASET == 'RUNMC':
            test_ids = np.array([15, 4, 6, 18, 14, 3]) # cases 11, 14, 16, 19, 21, 24
        elif args.PROMISE_SUB_DATASET == 'UCL':
            test_ids = np.array([11, 7, 1, 5, 16, 9]) # cases 1, 26, 29, 31, 34, 36
        elif args.PROMISE_SUB_DATASET == 'BIDMC':
            test_ids = np.array([8, 2, 19]) # cases 4, 6, 9
        elif args.PROMISE_SUB_DATASET == 'HK':
            test_ids = np.array([10, 13, 17, 12, 0]) # 39, 41, 44, 46, 49

    exp_str = exp_str + td_string

log_dir_tta = log_dir + exp_str
tensorboard_dir_tta = tensorboard_dir + exp_str

logging.info('Tensorboard directory TTA: %s' %tensorboard_dir_tta)

if not tf.gfile.Exists(log_dir_tta):
    tf.gfile.MakeDirs(log_dir_tta)
    tf.gfile.MakeDirs(tensorboard_dir_tta)

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
    # Define PDF matching loss
    # ================================================================
    # placeholder for SD PDFs (mean over all SD subjects). These will be extracted after loading the SD trained model.
    # The shapes have to be hard-coded. Can't get the tile operations to work otherwise..
    sd_pdf_pl = tf.placeholder(tf.float32, shape = [704, 61], name = 'sd_pdfs') # shape [num_channels, num_points_along_intensity_range]
    # placeholder for the standard deviation in the SD KDEs over the SD subjects.
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

    # D_KL (p_s, p_t) = \sum_{x} p_s(x) log( p_s(x) / p_t(x) )
    loss_all_kl_op = tf.reduce_mean(tf.reduce_sum(tf.math.multiply(sd_pdf_pl, tf.math.log(tf.math.divide(sd_pdf_pl, td_pdfs + 1e-5) + 1e-2)), axis = 1))

    # L2 distance between PDFs, with each coordinate scaled according to the log-variance across the SD subjects at that intensity value.
    epsilon = 1e-10
    loss_all_std_w1_op = tf.reduce_mean(tf.math.square(tf.math.divide(td_pdfs - sd_pdf_pl, 0.001 * sd_pdf_std_pl + 1e-3))) # mean over all channels of all layers
    loss_all_std_w2_op = tf.reduce_mean(tf.math.square(tf.math.divide(td_pdfs - sd_pdf_pl, tf.math.log(sd_pdf_std_pl + epsilon)))) # mean over all channels of all layers
    loss_all_std_w3_op = tf.reduce_mean(tf.math.square(tf.math.multiply(td_pdfs - sd_pdf_pl, 0.1 * tf.math.log(sd_pdf_std_pl + epsilon)))) # mean over all channels of all layers

    # compute means (across spatial locations and the batch axis) from the PDFs : $ \mu = \sum_{i=xmin}^{xmax} x * p(x) $
    x_pdf_tiled = tf.tile(tf.expand_dims(x_pdf_pl, 0), multiples = [td_pdfs.shape[0], 1]) # [Nc, Nx]
    td_pdf_means = tf.reduce_sum(tf.math.multiply(td_pdfs, x_pdf_tiled), axis = 1) # [Nc]
    sd_pdf_means = tf.reduce_sum(tf.math.multiply(sd_pdf_pl, x_pdf_tiled), axis = 1) # [Nc]
    loss_one_op = tf.reduce_mean(tf.math.square(td_pdf_means - sd_pdf_means)) # [Nc] (before reduce_mean)

    # compute variances (across spatial locations and the batch axis) from the PDFs, using the means computed above
    # $ \sigma^2 = \sum_{i=xmin}^{xmax} (x - \mu)^2 * p(x) $
    td_pdf_variances_tmp = tf.math.square(x_pdf_tiled - tf.tile(tf.expand_dims(td_pdf_means, 1), multiples = [1, x_pdf_tiled.shape[1]]))
    td_pdf_variances = tf.reduce_sum(tf.math.multiply(td_pdfs, td_pdf_variances_tmp), axis = 1) # [Nc]
    sd_pdf_variances_tmp = tf.math.square(x_pdf_tiled - tf.tile(tf.expand_dims(sd_pdf_means, 1), multiples = [1, x_pdf_tiled.shape[1]]))
    sd_pdf_variances = tf.reduce_sum(tf.math.multiply(sd_pdf_pl, sd_pdf_variances_tmp), axis = 1) # [Nc]
    loss_onetwo_op = tf.reduce_mean(tf.math.square(td_pdf_means - sd_pdf_means) + tf.math.square(td_pdf_variances - sd_pdf_variances))

    # D_KL (N(\mu_s, \sigma_s), N(\mu_t, \sigma_t)) = log(\sigma_t**2 / \sigma_s**2) + (\sigma_s**2 + (\mu_s - \mu_t)**2) / (\sigma_t**2)
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
    elif args.match_moments == 'all_kl': 
        loss_op = loss_all_kl_op
    # match the PDFs, with less weight for points where the variance over the SD subject is high
    elif args.match_moments == 'all_std': 
        loss_op = loss_all_std_w1_op
    # match the PDFs, with less weight for points where the variance over the SD subject is high
    elif args.match_moments == 'all_std_log': 
        loss_op = loss_all_std_w2_op
    # match the PDFs, with less weight for points where the variance over the SD subject is high
    elif args.match_moments == 'all_std_log_multiply': 
        loss_op = loss_all_std_w3_op
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
    tf.summary.scalar('loss/1D_all_kl', loss_all_kl_op)
    tf.summary.scalar('loss/1D_all_std_w1', loss_all_std_w1_op) # divide by std
    tf.summary.scalar('loss/1D_all_std', loss_all_std_w2_op) # divide by log-std
    tf.summary.scalar('loss/1D_all_std_log_multipled', loss_all_std_w3_op) # multiply with log-std
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
    saver_tta = tf.train.Saver(var_list = tta_vars, max_to_keep=10)   
    saver_tta_best = tf.train.Saver(var_list = tta_vars, max_to_keep=3)   
            
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

        pdfs_sd = np.array(pdfs_sd)

        # ================================================================
        # save
        # ================================================================
        np.save(sd_pdfs_filename, pdfs_sd) # [num_subjects, num_channels, num_x_points]

    # ================================================================
    # Determine which SD subject has the closest KDE to the current target image
    # (in terms of D_KL)
    # ================================================================
    if args.TTA_or_SFDA == 'TTA':
        if args.match_with_sd == 3 or args.match_with_sd == 4:
            
            kl_td_sd_subjects = np.zeros((pdfs_sd.shape[0]))
            num_runs_for_each_sd_subject = 10
            
            for sd_sub_num in range(pdfs_sd.shape[0]):
                
                kl_td_sd_subject = 0.0
                
                for _ in range(num_runs_for_each_sd_subject):

                    x_batch = test_image[np.random.randint(0, test_image.shape[0], args.b_size), :, :]

                    if args.match_moments == 'all_kl':
                        kl_td_sd_subject = kl_td_sd_subject + sess.run(loss_all_kl_op,
                                                                        feed_dict={images_pl: np.expand_dims(x_batch, axis=-1),
                                                                                    sd_pdf_pl: pdfs_sd[sd_sub_num, :, :], 
                                                                                    x_pdf_pl: x_values, 
                                                                                    alpha_pl: alpha})

                    elif args.match_moments == 'firsttwo_kl':
                        kl_td_sd_subject = kl_td_sd_subject + sess.run(loss_onetwokl_op,
                                                                        feed_dict={images_pl: np.expand_dims(x_batch, axis=-1),
                                                                                    sd_pdf_pl: pdfs_sd[sd_sub_num, :, :], 
                                                                                    x_pdf_pl: x_values, 
                                                                                    alpha_pl: alpha})
                                                                                    
                kl_td_sd_subjects[sd_sub_num] = kl_td_sd_subject / num_runs_for_each_sd_subject

            logging.info("D_KL with all SD subjects --> ")
            logging.info(kl_td_sd_subjects)
            sd_closest_sub = np.argsort(kl_td_sd_subjects)[0]
            logging.info('SD subject ' + str(sd_closest_sub) + ' is closest to this TD subject.')
            pdfs_sd_close = pdfs_sd[np.argsort(kl_td_sd_subjects)[0:6], :, :] # select 5 close subjects
    
    # ================================================================
    # TTA / SFDA iterations
    # ================================================================
    step = 0
    best_loss = 1000.0
    if args.TTA_or_SFDA == 'TTA':
        max_steps_i2i = exp_config.max_steps_i2i
    elif args.TTA_or_SFDA == 'SFDA':
        max_steps_i2i = num_test_subjects * exp_config.max_steps_i2i

    while (step < max_steps_i2i):
        
        logging.info("TTA / SFDA step: " + str(step+1))
        
        # run accumulated_gradients_zero_op (no need to provide values for any placeholders)
        sess.run(accumulated_gradients_zero_op)
        num_accumulation_steps = 0
        loss_this_step = 0.0

        # =============================
        # SD PDF to match with
        # =============================
        if args.match_with_sd == 1: # match with mean PDF over SD subjects
            sd_pdf_this_step = np.mean(pdfs_sd, axis = 0)
        elif args.match_with_sd == 2: # select a different SD subject for each TTA iteration
            sd_pdf_this_step = pdfs_sd[np.random.randint(pdfs_sd.shape[0]), :, :]
        elif args.match_with_sd == 3: # Match the target image's PDF to the closest SD PDF
            if args.TTA_or_SFDA == 'TTA':
                sd_pdf_this_step = pdfs_sd_close[0, :, :]
            else:
                logging.info("CANNOT FIND 'CLOSEST' SD subject while doing SFDA!")
        elif args.match_with_sd == 4: # Match the target image's PDF to one of the 5 closest SD PDFs (randomly chosen in each iteration)
            if args.TTA_or_SFDA == 'TTA':
                sd_pdf_this_step = pdfs_sd_close[np.random.randint(pdfs_sd_close.shape[0]), :, :]
            else:
                logging.info("CANNOT FIND 'CLOSEST 5' SD subjects while doing SFDA!")
                        
        # =============================
        # For SFDA, select a different TD subject in each adaptation epochs
        # =============================
        if args.TTA_or_SFDA == 'SFDA':
            sub_num = test_ids[np.random.randint(test_ids.shape[0])]
            subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])
            subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])
            test_image = imts[subject_id_start_slice:subject_id_end_slice,:,:]  
            test_image_gt = gtts[subject_id_start_slice:subject_id_end_slice,:,:]  
            test_image_gt = test_image_gt.astype(np.uint8)

        # =============================
        # Adaptation iterations within this epoch
        # =============================
        b_size = args.b_size
        for b_i in range(0, test_image.shape[0], b_size):

            if args.batch_randomized == 0:
                if b_i + b_size < test_image.shape[0]: # ignoring the rest of the image (that doesn't fit the last batch) for now.
                    # run the accumulate gradients op 
                    feed_dict={images_pl: np.expand_dims(test_image[b_i:b_i+b_size, :, :], axis=-1),
                                sd_pdf_pl: sd_pdf_this_step, 
                                sd_pdf_std_pl: np.std(pdfs_sd, axis = 0),
                                x_pdf_pl: x_values, 
                                alpha_pl: alpha}
                    sess.run(accumulate_gradients_op, feed_dict=feed_dict)
                    loss_this_step = loss_this_step + sess.run(loss_op, feed_dict = feed_dict)
                    num_accumulation_steps = num_accumulation_steps + 1

            elif args.batch_randomized == 1:      
                    # run the accumulate gradients op 
                    feed_dict={images_pl: np.expand_dims(test_image[np.random.randint(0, test_image.shape[0], b_size), :, :], axis=-1),
                                sd_pdf_pl: sd_pdf_this_step, 
                                sd_pdf_std_pl: np.std(pdfs_sd, axis = 0),
                                x_pdf_pl: x_values, 
                                alpha_pl: alpha}
                    sess.run(accumulate_gradients_op, feed_dict=feed_dict)
                    loss_this_step = loss_this_step + sess.run(loss_op, feed_dict = feed_dict)
                    num_accumulation_steps = num_accumulation_steps + 1

        loss_this_step = loss_this_step / num_accumulation_steps # average loss (over all slices of the image volume) in this step

        # ===========================
        # save best model so far
        # ===========================
        if args.TTA_or_SFDA == 'TTA':
            loss_for_deciding_best_model = loss_this_step
        elif args.TTA_or_SFDA == 'SFDA': # TODO change this to something like a rolling average loss over multiple steps
            loss_for_deciding_best_model = loss_this_step

        if best_loss > loss_for_deciding_best_model:
            best_loss = loss_for_deciding_best_model
            best_file = os.path.join(log_dir_tta, 'models/best_loss.ckpt')
            saver_tta_best.save(sess, best_file, global_step=step)
            logging.info('Found new best score (%f) at step %d -  Saving model.' % (best_loss, step))

        # ===========================
        # run accumulated_gradients_mean_op, with a value for the placeholder num_accumulation_steps_pl; followed by the train_op with applies the gradients
        # ===========================
        sess.run(accumulated_gradients_mean_op, feed_dict = {num_accumulation_steps_pl: num_accumulation_steps})
        # run the train_op. this also requires input output placeholders, as compute_gradients will be called again, but the returned gradient values will be replaced by the mean gradients.
        sess.run(train_op, feed_dict = feed_dict)

        # ===========================
        # Periodically save models
        # ===========================
        if (step+1) % 250 == 0:
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

        if args.test_dataset == 'PROMISE':
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
                                    np.mean(pdfs_sd, axis = 0),
                                    np.std(pdfs_sd, axis = 0),
                                    pdfs_td_this_step,
                                    x_values,
                                    log_dir_tta)

            # ===========================
            # visualize feature distribution alignment
            # ===========================
            b_i = 0
            sd_cfs_batch_this_step = sess.run(sd_cfs, feed_dict={sd_pdf_pl: np.mean(pdfs_sd, axis = 0)})
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
