import tensorflow as tf
from tfwrapper import losses
import matplotlib
import matplotlib.cm

# ================================================================
# ================================================================
def predict_i2l(images,
                exp_config,
                training_pl,
                nlabels,
                scope_reuse = False,
                return_features = False):
    '''
    Returns the prediction for an image given a network from the model zoo
    :param images: An input image tensor
    :param inference_handle: A model function from the model zoo
    :return: A prediction mask, and the corresponding softmax output
    '''

    if return_features == False:
        logits = exp_config.model_handle_i2l(images,
                                             nlabels = nlabels,
                                             training_pl = training_pl,
                                             scope_reuse = scope_reuse,
                                             return_features = return_features)

    else:
        logits, feat1, feat2, feat3 = exp_config.model_handle_i2l(images,
                                                                  nlabels = nlabels,
                                                                  training_pl = training_pl,
                                                                  scope_reuse = scope_reuse,
                                                                  return_features = return_features)
    
    softmax = tf.nn.softmax(logits)
    mask = tf.argmax(softmax, axis=-1)

    if return_features == False:
        return logits, softmax, mask
    else:
        return logits, softmax, mask, feat1, feat2, feat3

# ================================================================
# ================================================================
def predict_i2l_with_adaptors(images,
                              exp_config,
                              training_pl,
                              nlabels,
                              scope_reuse = False,
                              return_features = False):
    '''
    Similar to predict_i2l, but defines adaptors at different feature levels, that can be adapted for each test image
    '''

    if return_features == False:
        logits = exp_config.model_handle_i2l_with_adaptors(images,
                                                           nlabels = nlabels,
                                                           training_pl = training_pl,
                                                           return_features = return_features)
    else:
        logits, feat1, feat2, feat3 = exp_config.model_handle_i2l_with_adaptors(images,
                                                                                nlabels = nlabels,
                                                                                training_pl = training_pl,
                                                                                return_features = return_features)
    
    softmax = tf.nn.softmax(logits)
    mask = tf.argmax(softmax, axis=-1)

    if return_features == False:
        return logits, softmax, mask
    else:
        return logits, softmax, mask, feat1, feat2, feat3

# ================================================================
# ================================================================
def predict_dae(inputs,
                exp_config,
                nlabels,
                training_pl):

    logits = exp_config.model_handle_l2l(inputs,
                                         nlabels = nlabels,
                                         training_pl = training_pl)

    softmax = tf.nn.softmax(logits)
    mask = tf.argmax(softmax, axis=-1)

    return logits, softmax, mask

# ================================================================
# ================================================================
def autoencode(inputs,
               exp_config,
               training_pl,
               ae_features):

    recon = exp_config.model_handle_self_sup_ae(inputs,
                                                training_pl = training_pl,
                                                ae_features = ae_features)
    
    return recon

# ================================================================
# ================================================================
def predict_self_sup_vae(inputs,
                         exp_config,
                         training_pl):

    recon, latent_mu, latent_std = exp_config.model_handle_self_sup_vae(inputs,
                                                                        training_pl = training_pl)
    
    return recon, latent_mu, latent_std

# ================================================================
# ================================================================
def normalize(images,
              exp_config,
              training_pl,
              scope_reuse = False):
    
    images_normalized, added_residual = exp_config.model_handle_normalizer(images,
                                                                           exp_config,
                                                                           training_pl,
                                                                           scope_reuse = scope_reuse)
    
    return images_normalized, added_residual

# ================================================================
# ================================================================
def adapt_Ax(images, exp_config, instance_norm = 0):

    return exp_config.model_handle_adaptorAx(images, instance_norm)

# ================================================================
# ================================================================
def spectral_loss(w):

    return losses.spectral_norm(w[0,0,:,:])

# ================================================================
# ================================================================
def loss(logits,
         labels,
         nlabels,
         loss_type,
         mask_for_loss_within_mask = None,
         are_labels_1hot = False):
    '''
    Loss to be minimised by the neural network
    :param logits: The output of the neural network before the softmax
    :param labels: The ground truth labels in standard (i.e. not one-hot) format
    :param nlabels: The number of GT labels
    :param loss_type: Can be 'crossentropy'/'dice'/
    :return: The segmentation
    '''

    if are_labels_1hot is False:
        labels = tf.one_hot(labels, depth=nlabels)

    if loss_type == 'crossentropy':
        segmentation_loss = losses.pixel_wise_cross_entropy_loss(logits, labels)
        
    elif loss_type == 'crossentropy_reverse':
        predicted_probabilities = tf.nn.softmax(logits)
        segmentation_loss = losses.pixel_wise_cross_entropy_loss_using_probs(predicted_probabilities, labels)
        
    elif loss_type == 'dice':
        segmentation_loss = losses.dice_loss(logits, labels)
        
    elif loss_type == 'dice_within_mask':
        if mask_for_loss_within_mask is not None:
            segmentation_loss = losses.dice_loss_within_mask(logits, labels, mask_for_loss_within_mask)

    else:
        raise ValueError('Unknown loss: %s' % loss_type)

    return segmentation_loss

# ================================================================
# ================================================================
def likelihood_loss(pred_img_from_pred_seg_inverted,
                    img_orig,
                    loss_type):
    
    if loss_type is 'l2':
        loss_likelihood_op = tf.reduce_mean(tf.reduce_sum(tf.square(pred_img_from_pred_seg_inverted - img_orig), axis=[1,2,3]))
                
    elif loss_type is 'ssim':    
        loss_likelihood_op = 1 - tf.reduce_mean(tf.image.ssim(img1 = pred_img_from_pred_seg_inverted,
                                                              img2 = img_orig,
                                                              max_val = 1.0))
        
    return loss_likelihood_op

# ================================================================
# ================================================================
def training_step(loss,
                  var_list,
                  optimizer_handle,
                  learning_rate,
                  update_bn_nontrainable_vars = True,
                  return_optimizer = False):
    '''
    Creates the optimisation operation which is executed in each training iteration of the network
    :param loss: The loss to be minimised
    :var_list: list of params that this loss should be optimized wrt.
    :param optimizer_handle: A handle to one of the tf optimisers 
    :param learning_rate: Learning rate
    :return: The training operation
    '''

    optimizer = optimizer_handle(learning_rate = learning_rate) 
    train_op = optimizer.minimize(loss, var_list=var_list)
    
    if update_bn_nontrainable_vars is True:
        opt_memory_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        train_op = tf.group([train_op, opt_memory_update_ops])

    if return_optimizer is True:
        return train_op, optimizer
    else:
        return train_op

# ================================================================
# ================================================================
def evaluate_losses(logits,
                    labels,
                    nlabels,
                    loss_type,
                    are_labels_1hot = False):
    '''
    A function to compute various loss measures to compare the predicted and ground truth annotations
    '''
    
    # =================
    # supervised loss that is being optimized
    # =================
    supervised_loss = loss(logits = logits,
                           labels = labels,
                           nlabels = nlabels,
                           loss_type = loss_type,
                           are_labels_1hot = are_labels_1hot)

    # =================
    # per-structure dice for each label
    # =================
    if are_labels_1hot is False:
        labels = tf.one_hot(labels, depth=nlabels)
    dice_all_imgs_all_labels, mean_dice, mean_dice_fg = losses.compute_dice(logits, labels)
    
    return supervised_loss, dice_all_imgs_all_labels, mean_dice, mean_dice_fg

# ================================================================
# ================================================================
def evaluation_i2l(logits,
                   labels,
                   images,
                   nlabels,
                   loss_type):

    # =================
    # compute loss and foreground dice
    # =================
    supervised_loss, dice_all_imgs_all_labels, mean_dice, mean_dice_fg = evaluate_losses(logits,
                                                                                         labels,
                                                                                         nlabels,
                                                                                         loss_type)

    # =================
    # 
    # =================
    mask = tf.argmax(tf.nn.softmax(logits, axis=-1), axis=-1)
    mask_gt = labels
    
    # =================
    # write some segmentations to tensorboard
    # =================
    gt1 = prepare_tensor_for_summary(mask_gt, mode='mask', n_idx_batch=0, nlabels=nlabels)
    gt2 = prepare_tensor_for_summary(mask_gt, mode='mask', n_idx_batch=1, nlabels=nlabels)
    gt3 = prepare_tensor_for_summary(mask_gt, mode='mask', n_idx_batch=2, nlabels=nlabels)
    
    pred1 = prepare_tensor_for_summary(mask, mode='mask', n_idx_batch=0, nlabels=nlabels)
    pred2 = prepare_tensor_for_summary(mask, mode='mask', n_idx_batch=1, nlabels=nlabels)
    pred3 = prepare_tensor_for_summary(mask, mode='mask', n_idx_batch=2, nlabels=nlabels)
    
    img1 = prepare_tensor_for_summary(images, mode='image', n_idx_batch=0, nlabels=nlabels)
    img2 = prepare_tensor_for_summary(images, mode='image', n_idx_batch=1, nlabels=nlabels)
    img3 = prepare_tensor_for_summary(images, mode='image', n_idx_batch=2, nlabels=nlabels)
    
    tf.summary.image('example_labels_true', tf.concat([gt1, gt2, gt3], axis=0))
    tf.summary.image('example_labels_pred', tf.concat([pred1, pred2, pred3], axis=0))
    tf.summary.image('example_images', tf.concat([img1, img2, img3], axis=0))

    return supervised_loss, mean_dice

# ================================================================
# ================================================================
def evaluation_dae(clean_labels,
                   noisy_labels,
                   denoised_logits,
                   nlabels,
                   loss_type = 'dice',
                   are_labels_1hot = True):
    
    # =================
    # compute dice loss 
    # =================        
    # dice_loss = losses.dice_loss(denoised_logits, tf.one_hot(clean_labels, depth=nlabels))
    dice_loss = losses.dice_loss(denoised_logits, clean_labels)

    noisy_labels = tf.argmax(noisy_labels, axis=-1)
    denoised_labels = tf.argmax(tf.nn.softmax(denoised_logits, axis=-1), axis=-1)    

    if are_labels_1hot == False:
        clean_labels = clean_labels
    else:
        clean_labels = tf.argmax(clean_labels, axis=-1)
    
    # =================
    # write some segmentations to tensorboard
    # =================
    z = int(clean_labels.shape[1])
    
    z_idx = [z//2-4, z//2, z//2+4]
    
    gt1 = prepare_tensor_for_summary(clean_labels, mode='mask', n_idx_z=z_idx[0], nlabels=nlabels)
    gt2 = prepare_tensor_for_summary(clean_labels, mode='mask', n_idx_z=z_idx[1], nlabels=nlabels)
    gt3 = prepare_tensor_for_summary(clean_labels, mode='mask', n_idx_z=z_idx[2], nlabels=nlabels)
    
    noisy1 = prepare_tensor_for_summary(noisy_labels, mode='mask', n_idx_z=z_idx[0], nlabels=nlabels)
    noisy2 = prepare_tensor_for_summary(noisy_labels, mode='mask', n_idx_z=z_idx[1], nlabels=nlabels)
    noisy3 = prepare_tensor_for_summary(noisy_labels, mode='mask', n_idx_z=z_idx[2], nlabels=nlabels)
    
    pred1 = prepare_tensor_for_summary(denoised_labels, mode='mask', n_idx_z=z_idx[0], nlabels=nlabels)
    pred2 = prepare_tensor_for_summary(denoised_labels, mode='mask', n_idx_z=z_idx[1], nlabels=nlabels)
    pred3 = prepare_tensor_for_summary(denoised_labels, mode='mask', n_idx_z=z_idx[2], nlabels=nlabels)
    
    tf.summary.image('eg_1true_2noisy_3denoised_A', tf.concat([gt1, noisy1, pred1], axis=0))
    tf.summary.image('eg_1true_2noisy_3denoised_B', tf.concat([gt2, noisy2, pred2], axis=0))
    tf.summary.image('eg_1true_2noisy_3denoised_C', tf.concat([gt3, noisy3, pred3], axis=0))

    return dice_loss

# ================================================================
# ================================================================
def evaluate_ae(ae_input, ae_output):
    
    if ae_input.shape[-1] != 1:
        ae_input_vis = tf.expand_dims(ae_input[:,:,:,1], axis=-1)
        ae_output_vis = tf.expand_dims(ae_output[:,:,:,1], axis=-1)
    else:
        ae_input_vis = ae_input
        ae_output_vis = ae_output

    # =================
    # write some images to tensorboard
    # =================
    in1 = prepare_tensor_for_summary(ae_input_vis, mode='image', n_idx_batch=0)
    in2 = prepare_tensor_for_summary(ae_input_vis, mode='image', n_idx_batch=1)
    in3 = prepare_tensor_for_summary(ae_input_vis, mode='image', n_idx_batch=2)

    out1 = prepare_tensor_for_summary(ae_output_vis, mode='image', n_idx_batch=0)
    out2 = prepare_tensor_for_summary(ae_output_vis, mode='image', n_idx_batch=1)
    out3 = prepare_tensor_for_summary(ae_output_vis, mode='image', n_idx_batch=2)
    
    tf.summary.image('ae_inputs', tf.concat([in1, in2, in3], axis=0))
    tf.summary.image('ae_outputs', tf.concat([out1, out2, out3], axis=0))

    return tf.reduce_mean(tf.square(ae_output - ae_input))

# ================================================================
# ================================================================
def evaluation_self_sup_vae(recons,
                            images,
                            z_mu,
                            z_std,
                            lambda_kl):
    
    # =================
    # write some images to tensorboard
    # =================
    img1 = prepare_tensor_for_summary(images, mode='image', n_idx_batch=0)
    img2 = prepare_tensor_for_summary(images, mode='image', n_idx_batch=1)
    img3 = prepare_tensor_for_summary(images, mode='image', n_idx_batch=2)

    rec1 = prepare_tensor_for_summary(recons, mode='image', n_idx_batch=0)
    rec2 = prepare_tensor_for_summary(recons, mode='image', n_idx_batch=1)
    rec3 = prepare_tensor_for_summary(recons, mode='image', n_idx_batch=2)
    
    tf.summary.image('example_recons', tf.concat([rec1, rec2, rec3], axis=0))
    tf.summary.image('example_images', tf.concat([img1, img2, img3], axis=0))

    loss_recon = tf.reduce_mean(tf.square(recons - images))
    loss_kl = -0.5 * tf.reduce_mean(1 + z_std - tf.square(z_mu) - tf.exp(z_std)) # the vae enc predicts log sigma
    
    return loss_recon + lambda_kl * loss_kl

# ================================================================
# ================================================================
def prepare_tensor_for_summary(img,
                               mode,
                               n_idx_batch=0,
                               n_idx_z=60,
                               nlabels=None):
    '''
    Format a tensor containing imgaes or segmentation masks such that it can be used with
    tf.summary.image(...) and displayed in tensorboard. 
    :param img: Input image or segmentation mask
    :param mode: Can be either 'image' or 'mask. The two require slightly different slicing
    :param idx: Which index of a minibatch to display. By default it's always the first
    :param nlabels: Used for the proper rescaling of the label values. If None it scales by the max label.. 
    :return: Tensor ready to be used with tf.summary.image(...)
    '''

    if mode == 'mask':
        if img.get_shape().ndims == 3:
            V = tf.slice(img, (n_idx_batch, 0, 0), (1, -1, -1))
        elif img.get_shape().ndims == 4:
            V = tf.slice(img, (n_idx_batch, n_idx_z, 0, 0), (1, 1, -1, -1))
        elif img.get_shape().ndims == 5:
            V = tf.slice(img, (n_idx_batch, n_idx_z, 0, 0, 0), (1, 1, -1, -1, 1))
        else: raise ValueError('Dont know how to deal with input dimension %d' % (img.get_shape().ndims))

    elif mode == 'image':
        if img.get_shape().ndims == 3:
            V = tf.slice(img, (n_idx_batch, 0, 0), (1, -1, -1))
        elif img.get_shape().ndims == 4:
            V = tf.slice(img, (n_idx_batch, 0, 0, 0), (1, -1, -1, 1))
        elif img.get_shape().ndims == 5:
            V = tf.slice(img, (n_idx_batch, 0, 0, n_idx_z, 0), (1, -1, -1, 1, 1))
        else: raise ValueError('Dont know how to deal with input dimension %d' % (img.get_shape().ndims))

    else: raise ValueError('Unknown mode: %s. Must be image or mask' % mode)

    if mode=='image' or not nlabels:
        V -= tf.reduce_min(V)
        V /= tf.reduce_max(V)
    else:
        V /= (nlabels - 1)  # The largest value in a label map is nlabels - 1.

    V *= 255
    V = tf.cast(V, dtype=tf.uint8) # (1,224,224)
    V = tf.squeeze(V)
    V = tf.expand_dims(V, axis=0)
    
    # gather
    if mode == 'mask':
        cmap = 'viridis'
        cm = matplotlib.cm.get_cmap(cmap)
        colors = tf.constant(cm.colors, dtype=tf.float32)
        V = tf.gather(colors, tf.cast(V, dtype=tf.int32)) # (1,224,224,3)
        
    elif mode == 'image':
        V = tf.reshape(V, tf.stack((-1, tf.shape(img)[1], tf.shape(img)[2], 1))) # (1,224,224,1)
    
    return V

# ================================================================
# ================================================================
def divide_vars_into_groups(all_vars,
                            AEs = False,
                            DAE = False):
    
    i2l_vars = []
    normalization_vars = []
    bn_vars = []
    adapt_ax_vars = []
    adapt_af_vars = []

    if AEs == True:
        ae_xn_vars = []
        ae_y_vars = []
        ae_f1_vars = []
        ae_f2_vars = []
        ae_f3_vars = []

    if DAE == True:
        dae_vars = []
    
    for v in all_vars:
        
        var_name = v.name   
        
        if AEs == True:
            if 'self_sup_ae_xn' in var_name:
                ae_xn_vars.append(v)
            if 'self_sup_ae_y' in var_name:
                ae_y_vars.append(v)
            if 'self_sup_ae_f1' in var_name:
                ae_f1_vars.append(v)
            if 'self_sup_ae_f2' in var_name:
                ae_f2_vars.append(v)
            if 'self_sup_ae_f3' in var_name:
                ae_f3_vars.append(v)

        if DAE == True:
            if 'l2l_mapper' in var_name:
                dae_vars.append(v)

        if 'image_normalizer' in var_name:
            i2l_vars.append(v)
            normalization_vars.append(v)
        
        if 'beta' in var_name or 'gamma' in var_name: # assumes to batch normalization layers in Ax and Af
            bn_vars.append(v)
        
        if 'adaptAx' in var_name:
            adapt_ax_vars.append(v)
        
        if 'adaptAf' in var_name:
            adapt_af_vars.append(v)
        
        elif 'i2l_mapper' in var_name:
            i2l_vars.append(v)

    # assuming only one of DAE or AEs is true at once
    if AEs == True:
        return i2l_vars, normalization_vars, bn_vars, adapt_ax_vars, adapt_af_vars, ae_xn_vars, ae_y_vars, ae_f1_vars, ae_f2_vars, ae_f3_vars
    elif DAE == True:
        return i2l_vars, normalization_vars, bn_vars, adapt_ax_vars, adapt_af_vars, dae_vars
    else:
        return i2l_vars, normalization_vars, bn_vars, adapt_ax_vars, adapt_af_vars