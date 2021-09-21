import tensorflow as tf
import numpy as np
import logging

## ======================================================================
## ======================================================================
def compute_dice(logits, labels, epsilon=1e-10):
    '''
    Computes the dice score between logits and labels
    :param logits: Network output before softmax
    :param labels: ground truth label masks
    :param epsilon: A small constant to avoid division by 0
    :return: dice (per label, per image in the batch)
    '''

    with tf.name_scope('dice'):

        prediction = tf.nn.softmax(logits)
        intersection = tf.multiply(prediction, labels)
        
        reduction_axes = [1,2]        
        
        # compute area of intersection, area of GT, area of prediction (per image per label)
        tp = tf.reduce_sum(intersection, axis=reduction_axes) 
        tp_plus_fp = tf.reduce_sum(prediction, axis=reduction_axes) 
        tp_plus_fn = tf.reduce_sum(labels, axis=reduction_axes)

        # compute dice (per image per label)
        dice = 2 * tp / (tp_plus_fp + tp_plus_fn + epsilon)
        
        # =============================
        # if a certain label is missing in the GT of a certain image and also in the prediction,
        # dice[this_image,this_label] will be incorrectly computed as zero whereas it should be 1.
        # =============================
        
        # mean over all images in the batch and over all labels.
        mean_dice = tf.reduce_mean(dice)
        
        # mean over all images in the batch and over all foreground labels.
        mean_fg_dice = tf.reduce_mean(dice[:,1:])
        
    return dice, mean_dice, mean_fg_dice

## ======================================================================
## ======================================================================
def compute_dice_3d_without_batch_axis(prediction,
                                       labels,
                                       epsilon=1e-10):

    with tf.name_scope('dice_3d_without_batch_axis'):

        intersection = tf.multiply(prediction, labels)        
        
        reduction_axes = [0, 1, 2]                
        
        # compute area of intersection, area of GT, area of prediction (per image per label)
        tp = tf.reduce_sum(intersection, axis=reduction_axes) 
        tp_plus_fp = tf.reduce_sum(prediction, axis=reduction_axes) 
        tp_plus_fn = tf.reduce_sum(labels, axis=reduction_axes)
        
        # compute dice (per image per label)
        dice = 2 * tp / (tp_plus_fp + tp_plus_fn + epsilon)
        
        # mean over all images in the batch and over all labels.
        mean_fg_dice = tf.reduce_mean(dice[1:])
        
    return mean_fg_dice

## ======================================================================
## ======================================================================
def dice_loss(logits, labels):
    
    with tf.name_scope('dice_loss'):
        
        _, mean_dice, mean_fg_dice = compute_dice(logits, labels)
        
        # loss = 1 - mean_fg_dice
        loss = 1 - mean_dice

    return loss

## ======================================================================
## ======================================================================
def dice_loss_within_mask(logits, labels, mask):
    
    with tf.name_scope('dice_loss_within_mask'):
        
        _, mean_dice, mean_fg_dice = compute_dice(tf.math.multiply(logits, mask),
                                                  tf.math.multiply(labels, mask))
        
        # loss = 1 - mean_fg_dice
        loss = 1 - mean_dice

    return loss

## ======================================================================
## ======================================================================
def pixel_wise_cross_entropy_loss(logits, labels):

    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels))
    
    return loss

## ======================================================================
## ======================================================================
def pixel_wise_cross_entropy_loss_using_probs(predicted_probabilities, labels):

    labels_copy = np.copy(labels)
    
    # add a small number for log and normalize
    labels_copy = labels_copy + 1e-20
    labels_copy = labels_copy / tf.expand_dims(tf.reduce_sum(labels_copy, axis=-1), axis=-1)
    
    # compute cross-entropy 
    loss = - tf.reduce_mean(tf.reduce_sum(predicted_probabilities * tf.math.log(labels_copy), axis=-1))    
    
    return loss

## ======================================================================
# SPECTRAL NORM REGULARIZATION (PYTORCH IMPLEMENTATION)
# INTRODUCED IN https://github.com/VITA-Group/Orthogonality-in-CNNs/blob/master/SVHN/train.py
# USED in https://github.com/YufanHe/self-domain-adapted-network/blob/b68ed003729b10e311d46ad2438c8eb4a93da535/utils/util.py
## ======================================================================
# def l2_reg_ortho(mdl):
# 	l2_reg = None
# 	for W in mdl.parameters():
# 		if W.ndimension() < 2:
# 			continue
# 		else:   
# 			cols = W[0].numel()
# 			rows = W.shape[0]
# 			w1 = W.view(-1,cols)
# 			wt = torch.transpose(w1,0,1)
# 			m  = torch.matmul(wt,w1)
# 			ident = Variable(torch.eye(cols,cols))
# 			ident = ident.cuda()

# 			w_tmp = (m - ident)
# 			height = w_tmp.size(0)
# 			u = normalize(w_tmp.new_empty(height).normal_(0,1), dim=0, eps=1e-12)
# 			v = normalize(torch.matmul(w_tmp.t(), u), dim=0, eps=1e-12)
# 			u = normalize(torch.matmul(w_tmp, v), dim=0, eps=1e-12)
# 			sigma = torch.dot(u, torch.matmul(w_tmp, v))

# 			if l2_reg is None:
# 				l2_reg = (sigma)**2
# 			else:
# 				l2_reg = l2_reg + (sigma)**2
# 	return l2_reg

## ======================================================================
# TENSORFLOW IMPLEMENTATION of Spectral Norm Regularization
## ======================================================================
def spectral_norm(w):

    # input : w : weight matrix whose spectral norm is to be computed

    # compute (w_transpose*w - I)
    w_ = tf.linalg.matmul(tf.transpose(w), w) - tf.eye(tf.shape(w)[0])

    # sample a random vector u
    u_ = tf.random.normal([tf.shape(w_)[0], 1], 0.0, 1.0, dtype=tf.float32)
    # normalize
    u = u_ / tf.math.maximum(tf.norm(u_), 1e-12)

    # updates according to eqn 7 of https://arxiv.org/abs/1810.09102
    v_ = tf.linalg.matmul(tf.transpose(w_), u)
    v_norm = tf.sqrt(tf.reduce_sum(tf.square(tf.squeeze(v_))) + 1e-9) # using tf.norm gives NaNs, if the sum of squares is very close to zero, apparently.
    v = v_ / tf.math.maximum(v_norm, 1e-12)

    # updates according to eqn 7 of https://arxiv.org/abs/1810.09102
    u_ = tf.linalg.matmul(w_, v)
    u_norm = tf.sqrt(tf.reduce_sum(tf.square(tf.squeeze(u_))) + 1e-9)
    u = u_ / tf.math.maximum(u_norm, 1e-12)

    # now,  the spectral norm of w can be approximated as follows according https://arxiv.org/abs/1810.09102
    # not sure how this approximation is derived though..
    sigma = tf.reduce_sum(tf.multiply(u, tf.linalg.matmul(w_, v)))
    loss = (sigma)**2

    return loss