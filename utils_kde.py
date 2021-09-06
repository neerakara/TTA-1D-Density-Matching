import os
import numpy as np
import tensorflow as tf
import logging
import config.params as exp_config

# ==============================================
# ==============================================
def determine_zero_padding_this_depth(n, dx, dy):

    if n==1 or n==7 or n==8:
        dx_ = dx
        dy_ = dy

    elif n==2 or n==6:
        dx_ = dx // 2
        dy_ = dy // 2

    elif n==3 or n==5:
        dx_ = dx // 4
        dy_ = dy // 4

    elif n==4:
        dx_ = dx // 8
        dy_ = dy // 8
    
    return dx_, dy_

# ================================================
# Function to define ops for computing parameters of Gaussian distributions at each channel of each layer
# ================================================
def compute_1d_gaussian_parameters(feature_subsampling_factor,
                                   features_randomized,
                                   delta_x_pl,
                                   delta_y_pl):

    # ========
    # Ops for computing parameters of Gaussian distributions at each channel of each layer
    # ========
    means_1d = tf.zeros([1])
    variances_1d = tf.ones([1])

    # ========
    # Iterate through all channels of all layers
    # ========
    for conv_block in [1,2,3,4,5,6,7]:
        for conv_sub_block in [1,2]:
            conv_string = str(conv_block) + '_' + str(conv_sub_block)
            
            # ========
            # Extract features of this layer
            # ========
            # features = tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '/Conv2D:0') # BEFORE BN
            features = tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '_bn/FusedBatchNorm:0') # AFTER BN

            # ========
            # Compute the first two moments for all channels of this layer
            # ========
            this_layer_means, this_layer_variances = compute_first_two_moments(features,
                                                                               feature_subsampling_factor,
                                                                               features_randomized,
                                                                               block_num = conv_block,
                                                                               deltax = delta_x_pl,
                                                                               deltay = delta_y_pl)
            # ========
            # Concat to list
            # ========                                                                                         
            means_1d = tf.concat([means_1d, this_layer_means], 0)
            variances_1d = tf.concat([variances_1d, this_layer_variances], 0)

    # ========
    # Compute the parameters of the Gaussians at the softmax layer
    # ========
    features = tf.get_default_graph().get_tensor_by_name('i2l_mapper/pred_seg_soft:0')
    logging.info(features.shape)
    this_layer_means, this_layer_variances = compute_first_two_moments(features,
                                                                       feature_subsampling_factor,
                                                                       features_randomized,
                                                                       block_num = 8,
                                                                       deltax = delta_x_pl,
                                                                       deltay = delta_y_pl)
    means_1d = tf.concat([means_1d, this_layer_means], 0)
    variances_1d = tf.concat([variances_1d, this_layer_variances], 0)

    # Remove the 'dummy' first entry
    means1d = means_1d[1:]
    variances1d = variances_1d[1:]

    return means1d, variances1d

# ==============================================
# ==============================================
def compute_first_two_moments(features,
                              feature_subsampling_factor,
                              features_randomized,
                              cov = 'DIAG',
                              block_num = 1,
                              deltax = 0,
                              deltay = 0):

    # discard zero padding
    if deltax != 0 or deltay != 0:
        dx, dy = determine_zero_padding_this_depth(block_num, deltax, deltay)
        features = tf.gather(features, tf.range(start=dx, limit=tf.shape(features)[1]-dx, delta=1), axis=1)
        features = tf.gather(features, tf.range(start=dy, limit=tf.shape(features)[2]-dy, delta=1), axis=2)

    # Reshape to bring all those axes together where you want to take moments across
    features = tf.reshape(features, (-1, tf.shape(features)[-1]))

    # Subsample the feature maps to relieve the memory constraint and enable higher batch sizes
    if feature_subsampling_factor != 1:
        
        if features_randomized == 0:
            features = features[::feature_subsampling_factor, :]
        
        elif features_randomized == 1:
            # https://stackoverflow.com/questions/49734747/how-would-i-randomly-sample-pixels-in-tensorflow
            # https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/gather.md
            random_indices = tf.random.uniform(shape=[tf.shape(features)[0] // feature_subsampling_factor],
                                               minval=0,
                                               maxval=tf.shape(features)[0] - 1,
                                               dtype=tf.int32)
            features = tf.gather(features, random_indices, axis=0)

    if cov == 'DIAG':
        # Return first two moments of the computed features
        return tf.nn.moments(features, axes = [0])
    
    elif cov == 'FULL':
        means = tf.reduce_mean(features, axis = 0)        
        # https://stackoverflow.com/questions/47709854/how-to-get-covariance-matrix-in-tensorflow?rq=1
        means_ = tf.reduce_mean(features, axis=0, keepdims=True)
        mx = tf.matmul(tf.transpose(means_), means_)
        vx = tf.matmul(tf.transpose(features), features) / tf.cast(tf.shape(features)[0], tf.float32)
        covariance = vx - mx

        return means, variance

# ==============================================
# ==============================================
def compute_pairwise_potentials(features, potential_type):
    
    features_sobel_edges = tf.image.sobel_edges(features)

    if potential_type == 1: # GRADIENT L2 Norm SQUARED
        pairwise_potentials = tf.reduce_sum(tf.math.square(features_sobel_edges), axis=-1)

    if potential_type == 2: # GRADIENT L2 Norm 
        # having some numerical issue if the small number is not added before taking the square root
        pairwise_potentials = tf.math.sqrt(tf.reduce_sum(tf.math.square(features_sobel_edges), axis=-1) + 1e-5)

    if potential_type == 3: # GRADIENT L1 Norm
        pairwise_potentials = tf.reduce_sum(tf.math.abs(features_sobel_edges), axis=-1)

    return pairwise_potentials

# ==============================================
# Function to define ops for computing KDEs at each channel of each layer
# ==============================================
def compute_1d_kdes(feature_subsampling_factor, # subsampling factor
                    features_randomized, # whether to sample randomly or uniformly
                    delta_x_pl, # zero padding information
                    delta_y_pl,
                    x_pdf_g1_pl, # points where the KDEs have to be evaluated
                    x_pdf_g2_pl,
                    x_pdf_g3_pl,
                    alpha_pl): # smoothing parameter
    # ==================================
    # GROUP 1
    # ==================================
    kdes_g1 = tf.zeros([1, x_pdf_g1_pl.shape[0]]) # shape [num_channels, num_points_along_intensity_range]
    for conv_block in [1,2,3,4,5,6,7]:
        for conv_sub_block in [1,2]:
            if conv_block == 7 and conv_sub_block == 2:
                continue
            conv_string = str(conv_block) + '_' + str(conv_sub_block)
            features = tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + conv_string + '_bn/FusedBatchNorm:0')
            channel_kde_this_layer = compute_feature_kdes(features,
                                                          feature_subsampling_factor,
                                                          features_randomized,
                                                          x_pdf_g1_pl,
                                                          alpha_pl,
                                                          block_num = conv_block,
                                                          deltax = delta_x_pl,
                                                          deltay = delta_y_pl)
            kdes_g1 = tf.concat([kdes_g1, channel_kde_this_layer], 0)
    # Ignore the zeroth column that was added at the start of the loop
    kdes_g1 = kdes_g1[1:, :]

    # ==================================
    # GROUP 2 (layer 7_2)
    # ==================================
    features = tf.get_default_graph().get_tensor_by_name('i2l_mapper/conv' + str(7) + '_' + str(2) + '_bn/FusedBatchNorm:0')
    kdes_g2 = compute_feature_kdes(features,
                                   feature_subsampling_factor,
                                   features_randomized,
                                   x_pdf_g2_pl,
                                   alpha_pl,
                                   block_num = 7,
                                   deltax = delta_x_pl,
                                   deltay = delta_y_pl)

    # ==================================
    # GROUP 3 (soft probabilities)
    # ==================================
    # features = tf.get_default_graph().get_tensor_by_name('i2l_mapper/pred/Conv2D:0') # logits
    features = tf.get_default_graph().get_tensor_by_name('i2l_mapper/pred_seg_soft:0') # soft probabilities
    kdes_g3 = compute_feature_kdes(features,
                                   feature_subsampling_factor,
                                   features_randomized,
                                   x_pdf_g3_pl,
                                   alpha_pl,
                                   block_num = 8,
                                   deltax = delta_x_pl,
                                   deltay = delta_y_pl)

    return kdes_g1, kdes_g2, kdes_g3

# ==============================================   
# ==============================================
def compute_feature_kdes(features,
                         feature_subsampling_factor,
                         features_randomized,
                         x,
                         alpha,
                         block_num = 1,
                         deltax = 0,
                         deltay = 0):

    # discard zero padding
    if deltax != 0 or deltay != 0:
        dx, dy = determine_zero_padding_this_depth(block_num, deltax, deltay)
        features = tf.gather(features, tf.range(start=dx, limit=tf.shape(features)[1]-dx, delta=1), axis=1)
        features = tf.gather(features, tf.range(start=dy, limit=tf.shape(features)[2]-dy, delta=1), axis=2)

    # Reshape to bring all those axes together where you want to consider values as iid samples
    features = tf.reshape(features, (-1, tf.shape(features)[-1]))

    # for Batch size 2:
    # 1_1 (131072, 16), 1_2 (131072, 16), 2_1 (32768, 32), 2_2 (32768, 32)
    # 3_1 (8192, 64), 3_2 (8192, 64), 4_1 (2048, 128), 4_2 (2048, 128)
    # 5_1 (8192, 64), 5_2 (8192, 64), 6_1 (32768, 32), 6_2 (32768, 32)
    # 7_1 (131072, 16), 7_2 (131072, 16)

    # Subsample the feature maps to relieve the memory constraint and enable higher batch sizes
    if feature_subsampling_factor != 1:
        if features_randomized == 0:
            features = features[::feature_subsampling_factor, :]
        elif features_randomized == 1:
            # https://stackoverflow.com/questions/49734747/how-would-i-randomly-sample-pixels-in-tensorflow
            # https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/gather.md
            random_indices = tf.random.uniform(shape=[tf.shape(features)[0] // feature_subsampling_factor],
                                               minval=0,
                                               maxval=tf.shape(features)[0] - 1,
                                               dtype=tf.int32)
            features = tf.gather(features, random_indices, axis=0)

    features = tf.tile(tf.expand_dims(features, 0), multiples = [x.shape[0], 1, 1])
    x_tmp = tf.tile(tf.expand_dims(tf.expand_dims(x, -1), -1), multiples = [1, tf.shape(features)[1], tf.shape(features)[2]])

    # the 3 dimensions are : 
    # 1. the intensity values where the pdf is evaluated,
    # 2. all the features (the pixels along the 2 spatial dimensions as well as the batch dimension are considered 1D iid samples)
    # 3. the channels 
    channel_pdf_this_layer_td = tf.reduce_mean(tf.math.exp(- alpha * tf.math.square(x_tmp - features)), axis=1)
    channel_pdf_this_layer_td = tf.transpose(channel_pdf_this_layer_td)
    # at the end, we get 1 pdf (evaluated at the intensity values in x) per channel

    return channel_pdf_this_layer_td

# ==============================================
# ==============================================
def compute_pca_latent_kdes_tf(latents, # [num_active_patches, num_latent_dims] (latent dims of all channels stacked together)
                               z, # [range where to evaluate the KDE] [num_z_points]
                               alpha): # [smoothness param]

    # https://stackoverflow.com/questions/47537552/tensorflow-how-to-deal-with-dynamic-shape-trying-to-tile-and-concatenate-two-te
    dim = tf.shape(latents)[0]

    latents = tf.tile(tf.expand_dims(latents, 0), multiples = [z.shape[0], 1, 1]) # [num_z_points, num_active_patches, num_latent_dims]
    z_tmp = tf.tile(tf.expand_dims(tf.expand_dims(z, -1), -1), multiples = [1, dim, latents.shape[2]]) # [num_z_points, num_active_patches, num_latent_dims]

    # the 3 dimensions are : 
    # 1. the intensity values where the pdf is evaluated,
    # 2. all samples
    # 3. the channels 
    channel_pdf_this_layer_td = tf.reduce_mean(tf.math.exp(- alpha * tf.math.square(z_tmp - latents)), axis=1)
    channel_pdf_this_layer_td = tf.transpose(channel_pdf_this_layer_td) 
    # at the end, we get 1 pdf (evaluated at the intensity values in x) per channel

    return channel_pdf_this_layer_td # [num_latent_dims, num_z_points]

# ==============================================
# ==============================================
def sample_sd_points(pdfs_sd_this_step,
                     num_pts_lebesgue,
                     x_values):

    # sample x_values from this pdf - the log-ratio in KL-divergence will be computed at these points
    x_indices_lebesgue = np.zeros((pdfs_sd_this_step.shape[0], num_pts_lebesgue, 2))
    for c in range(pdfs_sd_this_step.shape[0]):
        sd_pdf_this_step_this_channel = pdfs_sd_this_step[c,:]
        sd_pdf_this_step_this_channel = sd_pdf_this_step_this_channel / np.sum(sd_pdf_this_step_this_channel)
        x_indices_lebesgue[c,:,0] = c
        x_indices_lebesgue[c,:,1] = np.random.choice(np.arange(x_values.shape[0]), num_pts_lebesgue, p=sd_pdf_this_step_this_channel)

    return x_indices_lebesgue.astype(np.uint8)

# ==============================================
# D_KL (p_s, p_t) = \sum_{x} p_s(x) log( p_s(x) / p_t(x) )
# ==============================================
def compute_kl_between_kdes(sd_pdfs,
                            td_pdfs,
                            x_indices_lebesgue = None,
                            order = 'SD_vs_TD'):

    if order == 'SD_vs_TD':
        # (via Riemann integral)
        if x_indices_lebesgue == None:
            loss_kl_op = tf.reduce_mean(tf.reduce_sum(tf.math.multiply(sd_pdfs,
                                                                       tf.math.log(tf.math.divide(sd_pdfs,
                                                                                                  td_pdfs + 1e-5) + 1e-2)), axis = 1))

        # (via Lebesgue integral)
        else:
            loss_kl_op = tf.reduce_mean(tf.reduce_sum(tf.math.log(tf.math.divide(tf.gather_nd(sd_pdfs, x_indices_lebesgue),
                                                                                 tf.gather_nd(td_pdfs, x_indices_lebesgue) + 1e-5) + 1e-2), axis = 1))

    elif order == 'TD_vs_SD':
        # (via Riemann integral)
        if x_indices_lebesgue == None:
            loss_kl_op = tf.reduce_mean(tf.reduce_sum(tf.math.multiply(td_pdfs,
                                                                       tf.math.log(tf.math.divide(td_pdfs,
                                                                                                  sd_pdfs + 1e-5) + 1e-2)), axis = 1))

        # (via Lebesgue integral)
        else:
            loss_kl_op = tf.reduce_mean(tf.reduce_sum(tf.math.log(tf.math.divide(tf.gather_nd(td_pdfs, x_indices_lebesgue),
                                                                                 tf.gather_nd(sd_pdfs, x_indices_lebesgue) + 1e-5) + 1e-2), axis = 1))

    return loss_kl_op

# ==============================================
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.wasserstein_distance.html#scipy.stats.wasserstein_distance
# https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.energy_distance.html#scipy.stats.energy_distance
# ==============================================
def compute_em_between_kdes(sd_pdfs,
                            td_pdfs,
                            order = 1):
    
    # compute cdfs
    sd_cdfs = tf.math.cumsum(sd_pdfs, axis=1)
    td_cdfs = tf.math.cumsum(td_pdfs, axis=1)

    # compute Earthmover loss
    if order == 1:
        loss_em_op = tf.reduce_mean(tf.reduce_sum(tf.math.abs(sd_cdfs - td_cdfs), axis=1))
    elif order == 2:
        loss_em_op = tf.reduce_mean(tf.math.sqrt(2*tf.reduce_sum(tf.math.square(sd_cdfs - td_cdfs), axis=1)))

    return loss_em_op

# ==============================================
# D_KL (p_s, p_t) = \sum_{x} p_s(x) log( p_s(x) / p_t(x) )
# ==============================================
def compute_kl_between_gaussian(mu_sd,
                                var_sd,
                                mu_td,
                                var_td,
                                order = 'SD_vs_TD'):

    if order == 'SD_vs_TD':
        loss_gaussian_kl_op = tf.reduce_mean(tf.math.log(var_td / var_sd) + (var_sd + (mu_sd - mu_td)**2) / (var_td + 1e-2))

    elif order == 'TD_vs_SD':
        loss_gaussian_kl_op = tf.reduce_mean(tf.math.log(var_sd / var_td) + (var_td + (mu_td - mu_sd)**2) / var_sd)

    return loss_gaussian_kl_op
                                                                                            
# ==============================================
# ==============================================
def compute_kde_losses(sd_pdfs,
                       td_pdfs,
                       x,
                       x_indices_lebesgue,
                       order):

    # ==================================
    # Match all moments with KL loss
    # ==================================
    loss_all_kl_op = compute_kl_between_kdes(sd_pdfs, td_pdfs, order = order)

    # ==================================
    # Match all moments with KL loss (via Lebesgue integral)
    # ==================================
    loss_all_kl_lebesgue_op = compute_kl_between_kdes(sd_pdfs, td_pdfs, x_indices_lebesgue = x_indices_lebesgue, order = order)

    # ==================================
    # Match first two moments with KL loss (via the KDEs)
    # (The means and variance estimators computed like this may have different behaviour that the mean and variance estimators directly computed from the samples.)
    # ==================================
    # compute means (across spatial locations and the batch axis) from the PDFs : $ \mu = \sum_{i=xmin}^{xmax} x * p(x) $
    x_tiled = tf.tile(tf.expand_dims(x, 0), multiples = [td_pdfs.shape[0], 1]) # [Nc, Nx]
    td_pdf_means = tf.reduce_sum(tf.math.multiply(td_pdfs, x_tiled), axis = 1) # [Nc]
    sd_pdf_means = tf.reduce_sum(tf.math.multiply(sd_pdfs, x_tiled), axis = 1) # [Nc]
    # compute variances (across spatial locations and the batch axis) from the PDFs, using the means computed above
    # $ \sigma^2 = \sum_{i=xmin}^{xmax} (x - \mu)^2 * p(x) $
    td_pdf_variances_tmp = tf.math.square(x_tiled - tf.tile(tf.expand_dims(td_pdf_means, 1), multiples = [1, x_tiled.shape[1]]))
    td_pdf_variances = tf.reduce_sum(tf.math.multiply(td_pdfs, td_pdf_variances_tmp), axis = 1) # [Nc]
    sd_pdf_variances_tmp = tf.math.square(x_tiled - tf.tile(tf.expand_dims(sd_pdf_means, 1), multiples = [1, x_tiled.shape[1]]))
    sd_pdf_variances = tf.reduce_sum(tf.math.multiply(sd_pdfs, sd_pdf_variances_tmp), axis = 1) # [Nc]
    # D_KL (N(\mu_s, \sigma_s), N(\mu_t, \sigma_t)) = log(\sigma_t**2 / \sigma_s**2) + (\sigma_s**2 + (\mu_s - \mu_t)**2) / (\sigma_t**2)
    loss_gaussian_kl_op = tf.reduce_mean(tf.math.log(td_pdf_variances / sd_pdf_variances) + (sd_pdf_variances + (sd_pdf_means - td_pdf_means)**2) / td_pdf_variances)

    # ==================================
    # Match Full PDFs by minimizing the L2 distance between the corresponding Characteristic Functions (complex space)
    # ==================================
    # compute CFs of the source and target domains
    td_cfs = tf.spectral.rfft(td_pdfs)
    sd_cfs = tf.spectral.rfft(sd_pdfs)
    loss_all_cf_l2_op = tf.reduce_mean(tf.math.abs(td_cfs - sd_cfs)) # mean over all channels of all layers and all frequencies

    return loss_all_kl_op, loss_all_kl_lebesgue_op, loss_gaussian_kl_op, loss_all_cf_l2_op, sd_cfs, td_cfs

# ==================================
# ==================================
def compute_sd_kdes(train_dataset,
                    imtr,
                    image_depth_tr,
                    b_size,
                    sess,
                    images_pl,
                    alpha_pl,
                    alpha,
                    kdes_g1, kdes_g2, kdes_g3,
                    x_kde_g1_pl, x_kde_g2_pl, x_kde_g3_pl,
                    kde_g1_params, kde_g2_params, kde_g3_params,
                    res_x,
                    res_y,
                    target_res,
                    size_x,
                    size_y,
                    size_z,
                    delta_x_pl,
                    delta_y_pl,
                    savepath):

    # make array of points where the KDEs have to be evaluated
    x_values_g1 = np.arange(kde_g1_params[0], kde_g1_params[1] + kde_g1_params[2], kde_g1_params[2])
    x_values_g2 = np.arange(kde_g2_params[0], kde_g2_params[1] + kde_g2_params[2], kde_g2_params[2])
    x_values_g3 = np.arange(kde_g3_params[0], kde_g3_params[1] + kde_g3_params[2], kde_g3_params[2])

    kdes_sd_g1 = []
    kdes_sd_g2 = []
    kdes_sd_g3 = []

    # =========================
    # Loop over all SD subjects
    # =========================
    if train_dataset == 'HCPT1':
        num_subjects = 10 # 10 subjects should likely provide enough variability for brain 
    else:
        num_subjects = size_z.shape[0]

    for sub_num in range(num_subjects):

        # =========================
        # Extract one SD subject
        # =========================
        logging.info("==== Computing pdf for subject " + str(sub_num) + '..')
        if train_dataset == 'HCPT1': # circumventing a bug in the way size_z is written for HCP images
            sd_image = imtr[sub_num*image_depth_tr : (sub_num+1)*image_depth_tr,:,:]
        else:
            sd_image = imtr[np.sum(size_z[:sub_num]) : np.sum(size_z[:sub_num+1]),:,:]
        logging.info(sd_image.shape)

        # =========================
        # Compute the padding in this subject
        # =========================
        nxhat = size_x[sub_num] * res_x[sub_num] / target_res[0]
        nyhat = size_y[sub_num] * res_y[sub_num] / target_res[1]
        padding_x = np.maximum(0, sd_image.shape[1] - nxhat.astype(np.uint16)) // 2
        padding_y = np.maximum(0, sd_image.shape[2] - nyhat.astype(np.uint16)) // 2

        logging.info('shape after preproc: ' + str(sd_image.shape))
        logging.info('x-dim orig: ' + str(size_x[sub_num]))
        logging.info('x-res orig: ' + str(res_x[sub_num]))
        logging.info('size after resampling: ' + str(nxhat))
        logging.info('padding: ' + str(padding_x))
        
        # =========================
        # Loop over batches of this subject
        # =========================
        num_batches = 0
        for b_i in range(0, sd_image.shape[0], b_size):
            if b_i + b_size < sd_image.shape[0]: # ignoring the rest of the image (that doesn't fit the last batch) for now.
                
                kdes_batch_g1, kdes_batch_g2, kdes_batch_g3 = sess.run([kdes_g1, kdes_g2, kdes_g3],
                                                                       feed_dict={images_pl: np.expand_dims(sd_image[b_i:b_i+b_size, ...], axis=-1),
                                                                                  alpha_pl: alpha,
                                                                                  x_kde_g1_pl: x_values_g1,
                                                                                  x_kde_g2_pl: x_values_g2,
                                                                                  x_kde_g3_pl: x_values_g3,
                                                                                  delta_x_pl: padding_x,
                                                                                  delta_y_pl: padding_y})
                if b_i == 0:
                    kdes_this_subject_g1 = kdes_batch_g1
                    kdes_this_subject_g2 = kdes_batch_g2
                    kdes_this_subject_g3 = kdes_batch_g3
                else:
                    kdes_this_subject_g1 = kdes_this_subject_g1 + kdes_batch_g1
                    kdes_this_subject_g2 = kdes_this_subject_g2 + kdes_batch_g2
                    kdes_this_subject_g3 = kdes_this_subject_g3 + kdes_batch_g3
                
                num_batches = num_batches + 1
        
        kdes_this_subject_g1 = kdes_this_subject_g1 / num_batches
        kdes_this_subject_g2 = kdes_this_subject_g2 / num_batches
        kdes_this_subject_g3 = kdes_this_subject_g3 / num_batches
        
        # append to list 
        kdes_sd_g1.append(kdes_this_subject_g1)
        kdes_sd_g2.append(kdes_this_subject_g2)
        kdes_sd_g3.append(kdes_this_subject_g3)

    kdes_sd_g1 = np.array(kdes_sd_g1)
    kdes_sd_g2 = np.array(kdes_sd_g2)
    kdes_sd_g3 = np.array(kdes_sd_g3)

    # make the filenames where the KDEs have to be stored
    sd_kdes_fname_g1 = savepath + exp_config.make_sd_kde_name(b_size, alpha, kde_g1_params) + '_g1.npy'
    sd_kdes_fname_g2 = savepath + exp_config.make_sd_kde_name(b_size, alpha, kde_g2_params) + '_g2.npy'
    sd_kdes_fname_g3 = savepath + exp_config.make_sd_kde_name(b_size, alpha, kde_g3_params) + '_g3.npy'
    
    # save
    np.save(sd_kdes_fname_g1, kdes_sd_g1)
    np.save(sd_kdes_fname_g2, kdes_sd_g2)
    np.save(sd_kdes_fname_g3, kdes_sd_g3)

    return kdes_sd_g1, kdes_sd_g2, kdes_sd_g3

# ==================================
# ==================================
def compute_sd_gaussians(filename,
                         train_dataset,
                         imtr,
                         image_depth_tr,
                         b_size,
                         sess,
                         gaussian_mu,
                         gaussian_var,
                         images_pl,
                         res_x,
                         res_y,
                         target_res,
                         size_x,
                         size_y,
                         size_z,
                         delta_x_pl,
                         delta_y_pl):

    if os.path.isfile(filename):   
        logging.info("Gaussian params already computed. Loading..")         
        gaussians_sd = np.load(filename) # [num_subjects, num_channels, 2]

    else:
        logging.info("Computing Gaussian params..")         
        gaussians_sd = []            

        if train_dataset == 'HCPT1':
            num_subjects = 10 # 10 subjects should likely provide enough variability for brain 
        else:
            num_subjects = size_z.shape[0]

        for sub_num in range(num_subjects):
            
            logging.info("==== Computing Gaussian for subject " + str(sub_num) + '..')
            if train_dataset == 'HCPT1': # circumventing a bug in the way size_z is written for HCP images
                sd_image = imtr[sub_num*image_depth_tr : (sub_num+1)*image_depth_tr,:,:]
            else:
                sd_image = imtr[np.sum(size_z[:sub_num]) : np.sum(size_z[:sub_num+1]),:,:]
                        
            # =========================
            # Compute the padding in this subject
            # =========================
            nxhat = size_x[sub_num] * res_x[sub_num] / target_res[0]
            nyhat = size_y[sub_num] * res_y[sub_num] / target_res[1]
            padding_x = np.maximum(0, sd_image.shape[1] - nxhat.astype(np.uint16)) // 2
            padding_y = np.maximum(0, sd_image.shape[2] - nyhat.astype(np.uint16)) // 2

            logging.info('shape after preproc: ' + str(sd_image.shape))
            logging.info('x-dim orig: ' + str(size_x[sub_num]))
            logging.info('x-res orig: ' + str(res_x[sub_num]))
            logging.info('size after resampling: ' + str(nxhat))
            logging.info('padding: ' + str(padding_x))
            
            # =========================
            # If b_size != 0, Do batchwise computations
            # =========================
            if b_size != 0:
                num_batches = 0
                for b_i in range(0, sd_image.shape[0], b_size):
                    if b_i + b_size < sd_image.shape[0]: # ignoring the rest of the image (that doesn't fit the last batch) for now.                    
                        b_mu, b_var = sess.run([gaussian_mu, gaussian_var],
                                               feed_dict={images_pl: np.expand_dims(sd_image[b_i:b_i+b_size, ...], axis=-1),
                                                          delta_x_pl: padding_x,
                                                          delta_y_pl: padding_y})
                        if b_i == 0:
                            s_mu = b_mu
                            s_var = b_var
                        else:
                            s_mu = s_mu + b_mu
                            s_var = s_var + b_var
                        num_batches = num_batches + 1
                s_mu = s_mu / num_batches
                s_var = s_var / num_batches
            # =========================
            # Else, use full images to compute stats
            # =========================
            elif b_size == 0:
                s_mu, s_var = sess.run([gaussian_mu, gaussian_var],
                                       feed_dict={images_pl: np.expand_dims(sd_image, axis=-1),
                                                  delta_x_pl: padding_x,
                                                  delta_y_pl: padding_y})

            # Append to list
            gaussians_sd.append(np.stack((s_mu, s_var), 1))

        gaussians_sd = np.array(gaussians_sd)
        # save
        np.save(filename, gaussians_sd)

    return gaussians_sd

# ==================================
# ==================================
def extract_patches(features,
                    channel = 0,
                    psize = 64,
                    stride = 10):

    # features will be of shape: b_size, nx, ny, n_channels
    # let's treat all channels separately for now.
    # let's just select one channel for now

    patches = tf.image.extract_image_patches(tf.expand_dims(tf.gather(features, channel, axis=-1), axis=-1),
                                             ksizes = [1, psize, psize, 1],
                                             strides = [1, stride, stride, 1],
                                             rates = [1, 1, 1, 1],
                                             padding = 'VALID')

    patches_reshaped = tf.reshape(patches, [-1, patches.shape[-1]])

    return patches_reshaped

# ==================================
# ==================================
def compute_pca_latents(patches, mean, pcs, var):

    patches_centered = patches - mean
    latents = tf.matmul(patches_centered, tf.transpose(pcs))
    latents_whitened = latents / tf.sqrt(var)

    return latents_whitened

# ==================================
# ==================================
def compute_pca_latent_kdes(latents, alpha):

    z_min = -5.0
    z_max = 5.0
    res = 0.1
    z_vals = np.arange(z_min, z_max + res, res)

    kdes_this_subject = []
    for k in range(latents.shape[1]):
        z_samples = latents[:, k]
        kde_this_dim = np.mean(np.exp(-alpha * np.square(np.tile(z_vals, [z_samples.shape[0], 1]) - np.tile(z_samples, [z_vals.shape[0], 1]).T)), 0)
        kdes_this_subject.append(kde_this_dim)

    return np.array(kdes_this_subject), z_vals

# ==================================
# ==================================
def compute_latent_kdes_subjectwise(images,
                                    train_dataset,
                                    image_size,
                                    image_depths,
                                    image_placeholder,
                                    channel_placeholder,
                                    channel_num,
                                    features,
                                    patches,
                                    fg_probs,
                                    psize,
                                    threshold,
                                    learned_pca,
                                    alpha_kde,
                                    sess,
                                    savepath = ''):

    actpats_allsubs = np.zeros([1, psize*psize])
    feats_allsubs = np.zeros([image_depths.shape[0], image_size[0], image_size[1]]) 
    kdes_allsubs = []

    if train_dataset == 'HCPT1':
        num_subjects = 10 # 10 subjects should likely provide enough variability for brain 
    else:
        num_subjects = image_depths.shape[0]

    for sub_num in range(num_subjects):
        
        logging.info("Subject " + str(sub_num+1))
        
        # extract one subject
        image = images[np.sum(image_depths[:sub_num]) : np.sum(image_depths[:sub_num+1]),:,:]
        feed_dict={image_placeholder: np.expand_dims(image, axis=-1),
                   channel_placeholder: channel_num}
        
        # extract features from layer 7_2 for this subject
        features_this_sub = sess.run(features, feed_dict = feed_dict)
        feats_allsubs[sub_num, :, :] = features_this_sub[features_this_sub.shape[0]//2, :, :, channel_num]
        
        # extract patches from layer 7_2 channel 'channel' for this subject
        patches_this_sub = sess.run(patches, feed_dict = feed_dict)
        
        # extract predicted fg probs for this subject
        fg_probs_this_sub = sess.run(fg_probs, feed_dict=feed_dict)
        
        # keep only 'active' patches
        active_patches_this_sub = patches_this_sub[np.where(fg_probs_this_sub[:, (psize * (psize + 1))//2] > threshold)[0], :]
        actpats_allsubs = np.concatenate((actpats_allsubs, active_patches_this_sub), axis=0)
        
        # transform active patches to their latent representation
        active_patches_this_sub_latent = learned_pca.transform(active_patches_this_sub)
        
        # logging.info("Min latent value: " + str(np.round(np.min(active_patches_this_sub_latent), 2)))
        # logging.info("Max latent value: " + str(np.round(np.max(active_patches_this_sub_latent), 2)))
        
        # compute KDEs for each latent dimension for this subject
        kdes_this_sub, z_vals = compute_pca_latent_kdes(active_patches_this_sub_latent, alpha_kde)
        kdes_allsubs.append(kdes_this_sub)
    
    # save the KDEs of all SD training subjects for all latent dimensions, for feature channel 'channel'.
    kdes_allsubs = np.array(kdes_allsubs)
    
    if savepath != '':
        np.save(savepath, kdes_allsubs)
    
    return kdes_allsubs, z_vals, feats_allsubs, np.array(actpats_allsubs[1:,:])

# ==================================
# ==================================
def compute_latent_gaussians_subjectwise(images,
                                         train_dataset,
                                         image_size,
                                         image_depths,
                                         image_placeholder,
                                         channel_placeholder,
                                         channel_num,
                                         features,
                                         patches,
                                         fg_probs,
                                         psize,
                                         threshold,
                                         learned_pca,
                                         sess,
                                         savepath = ''):

    gaussians_allsubs = []

    if train_dataset == 'HCPT1':
        num_subjects = 10 # 10 subjects should likely provide enough variability for brain 
    else:
        num_subjects = image_depths.shape[0]

    for sub_num in range(num_subjects):
        
        logging.info("Subject " + str(sub_num+1))
        
        # extract one subject
        image = images[np.sum(image_depths[:sub_num]) : np.sum(image_depths[:sub_num+1]),:,:]
        feed_dict={image_placeholder: np.expand_dims(image, axis=-1),
                   channel_placeholder: channel_num}
                
        # extract patches from layer 7_2 channel 'channel' for this subject
        patches_this_sub = sess.run(patches, feed_dict = feed_dict)
        
        # extract predicted fg probs for this subject
        fg_probs_this_sub = sess.run(fg_probs, feed_dict=feed_dict)
        
        # keep only 'active' patches
        active_patches_this_sub = patches_this_sub[np.where(fg_probs_this_sub[:, (psize * (psize + 1))//2] > threshold)[0], :]
        
        # transform active patches to their latent representation
        latents_of_active_patches_of_this_sub = learned_pca.transform(active_patches_this_sub) # [num_patches, num_latent_dims]
       
        # compute Gaussian parameters for each latent dimension for this subject
        latent_means_this_sub = np.mean(latents_of_active_patches_of_this_sub, axis=0) # [num_latent_dims]
        latent_vars_this_sub = np.var(latents_of_active_patches_of_this_sub, axis=0) # [num_latent_dims]

        # append to list
        gaussians_allsubs.append(np.stack((latent_means_this_sub, latent_vars_this_sub), 1))

    # save
    gaussians_allsubs = np.array(gaussians_allsubs)
    logging.info(gaussians_allsubs.shape)
    if savepath != '':
        np.save(savepath, gaussians_allsubs)
    
    return gaussians_allsubs

# ==================================
# NUMPY. Riemann integral
# ==================================
def compute_kl_between_kdes_numpy(sd_pdfs, # [num_subjects, num_dims, num_evals_of_each_kde]
                                  td_pdfs): # [num_subjects, num_dims, num_evals_of_each_kde]

    kl1 = 0 # sd vs td
    kl2 = 0 # td vs sd

    for i in range(sd_pdfs.shape[0]):
        for j in range(td_pdfs.shape[0]):

            sd_pdf = sd_pdfs[i, :, :]
            td_pdf = td_pdfs[j, :, :]

            # (via Riemann integral)
            kl1 = kl1 + np.mean(np.sum(np.multiply(sd_pdf, np.log(np.divide(sd_pdf, td_pdf + 1e-5) + 1e-2)), axis = -1))
            kl2 = kl2 + np.mean(np.sum(np.multiply(td_pdf, np.log(np.divide(td_pdf, sd_pdf + 1e-5) + 1e-2)), axis = -1))

    avg_kl1 = kl1 / (sd_pdfs.shape[0] * td_pdfs.shape[0])
    avg_kl2 = kl2 / (sd_pdfs.shape[0] * td_pdfs.shape[0])

    return np.round(avg_kl1, 2), np.round(avg_kl2, 2)