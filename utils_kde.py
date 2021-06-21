import os
import numpy as np
import tensorflow as tf
import logging

# ==============================================
# ==============================================
def compute_first_two_moments(features,
                              feature_subsampling_factor,
                              features_randomized,
                              cov = 'DIAG'):

    # Reshape to bring all those axes together where you want to take moments across
    features = tf.reshape(features, (-1, features.shape[-1]))

    # Subsample the feature maps to relieve the memory constraint and enable higher batch sizes
    if feature_subsampling_factor != 1:
        
        if features_randomized == 0:
            features = features[::feature_subsampling_factor, :]
        
        elif features_randomized == 1:
            # https://stackoverflow.com/questions/49734747/how-would-i-randomly-sample-pixels-in-tensorflow
            # https://github.com/tensorflow/docs/blob/r1.12/site/en/api_docs/python/tf/gather.md
            random_indices = tf.random.uniform(shape=[features.shape[0].value // feature_subsampling_factor],
                                               minval=0,
                                               maxval=features.shape[0].value - 1,
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
# ==============================================
def compute_feature_kdes(features,
                         feature_subsampling_factor,
                         features_randomized,
                         x,
                         alpha):

    # Reshape to bring all those axes together where you want to consider values as iid samples
    features = tf.reshape(features, (-1, features.shape[-1]))

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
            random_indices = tf.random.uniform(shape=[features.shape[0].value // feature_subsampling_factor],
                                                minval=0,
                                                maxval=features.shape[0].value - 1,
                                                dtype=tf.int32)
            features = tf.gather(features, random_indices, axis=0)

    features = tf.tile(tf.expand_dims(features, 0), multiples = [x.shape[0], 1, 1])
    x_tmp = tf.tile(tf.expand_dims(tf.expand_dims(x, -1), -1), multiples = [1, features.shape[1], features.shape[2]])

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
                            x_indices_lebesgue = None):

    # (via Riemann integral)
    if x_indices_lebesgue == None:
        loss_kl_op = tf.reduce_mean(tf.reduce_sum(tf.math.multiply(sd_pdfs,
                                                                   tf.math.log(tf.math.divide(sd_pdfs,
                                                                                              td_pdfs + 1e-5) + 1e-2)), axis = 1))

    # (via Lebesgue integral)
    else:
        loss_kl_op = tf.reduce_mean(tf.reduce_sum(tf.math.log(tf.math.divide(tf.gather_nd(sd_pdfs, x_indices_lebesgue),
                                                                             tf.gather_nd(td_pdfs, x_indices_lebesgue) + 1e-5) + 1e-2), axis = 1))

    return loss_kl_op
                                                                                            
# ==============================================
# ==============================================
def compute_kde_losses(sd_pdfs,
                       td_pdfs,
                       x,
                       x_indices_lebesgue):

    # ==================================
    # Match all moments with KL loss
    # ==================================
    loss_all_kl_op = compute_kl_between_kdes(sd_pdfs, td_pdfs)

    # ==================================
    # Match all moments with KL loss (via Lebesgue integral)
    # ==================================
    loss_all_kl_lebesgue_op = compute_kl_between_kdes(sd_pdfs, td_pdfs, x_indices_lebesgue)

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
def compute_sd_pdfs(filename,
                    train_dataset,
                    imtr,
                    image_depth_tr,
                    orig_data_siz_z_train,
                    b_size,
                    sess,
                    td_pdfs,
                    images_pl,
                    x_pdf_pl,
                    x_values,
                    alpha_pl,
                    alpha):

    if os.path.isfile(filename):            
        pdfs_sd = np.load(filename) # [num_subjects, num_channels, 2]

    else:
        pdfs_sd = []
        for train_sub_num in range(orig_data_siz_z_train.shape[0]):
            logging.info("==== Computing pdf for subject " + str(train_sub_num) + '..')
            if train_dataset == 'HCPT1': # circumventing a bug in the way orig_data_siz_z_train is written for HCP images
                sd_image = imtr[train_sub_num*image_depth_tr : (train_sub_num+1)*image_depth_tr,:,:]
            else:
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
        # save
        np.save(filename, pdfs_sd)

    return pdfs_sd

# ==================================
# ==================================
def compute_sd_gaussians(filename,
                         train_dataset,
                         imtr,
                         image_depth_tr,
                         orig_data_siz_z_train,
                         b_size,
                         sess,
                         td_mu,
                         td_var,
                         images_pl):

    if os.path.isfile(filename):            
        gaussians_sd = np.load(filename) # [num_subjects, num_channels, 2]

    else:
        gaussians_sd = []            

        for train_sub_num in range(orig_data_siz_z_train.shape[0]):
            
            logging.info("==== Computing Gaussian for subject " + str(train_sub_num) + '..')
            if train_dataset == 'HCPT1': # circumventing a bug in the way orig_data_siz_z_train is written for HCP images
                sd_image = imtr[train_sub_num*image_depth_tr : (train_sub_num+1)*image_depth_tr,:,:]
            else:
                sd_image = imtr[np.sum(orig_data_siz_z_train[:train_sub_num]) : np.sum(orig_data_siz_z_train[:train_sub_num+1]),:,:]
            logging.info(sd_image.shape)
            
            # =========================
            # Do batchwise computations
            # =========================
            if b_size != 0:
                num_batches = 0
                for b_i in range(0, sd_image.shape[0], b_size):
                    if b_i + b_size < sd_image.shape[0]: # ignoring the rest of the image (that doesn't fit the last batch) for now.                    
                        b_mu, b_var = sess.run([td_mu, td_var], feed_dict={images_pl: np.expand_dims(sd_image[b_i:b_i+b_size, ...], axis=-1)})
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
            # Use full images to compute stats
            # =========================
            elif b_size == 0:
                s_mu, s_var = sess.run([td_mu, td_var], feed_dict={images_pl: np.expand_dims(sd_image, axis=-1)})

            # Append to list
            gaussians_sd.append(np.stack((s_mu, s_var), 1))

        gaussians_sd = np.array(gaussians_sd)
        # save
        np.save(filename, gaussians_sd)

    return gaussians_sd