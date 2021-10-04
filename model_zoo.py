# ======================================================================
# import stuff
# ======================================================================
import tensorflow as tf
from tfwrapper import layers

# ======================================================================
# adaptor Ax
# ======================================================================
def adaptor_Ax(images, instance_normalze = 0):
        
    with tf.variable_scope('adaptAx') as scope:
                                
        # default TF init is 'glorot_uniform' (https://stackoverflow.com/questions/43284047/what-is-the-default-kernel-initializer-in-tf-layers-conv2d-and-tf-layers-dense)
        # Yufan He MedIA 2021 use He Normal initializer (https://www.tensorflow.org/api_docs/python/tf/keras/initializers/HeNormal)

        adapt_f1 = tf.layers.conv2d(inputs=images,
                                    filters=64,
                                    kernel_size=1,
                                    padding='SAME',
                                    name='Axconv1',
                                    use_bias=True,
                                    activation=tf.identity,
                                    kernel_initializer = tf.initializers.random_normal(mean=1.0/1.0, stddev=tf.math.sqrt(2.0 / 1.0))) # initializing with mean weights around 1 to have close to an identity mapping at the start of TTA
        if instance_normalze == 1:
            adapt_f1 = layers.instance_normalize(adapt_f1) # using IN leads to very poor performance at the start of TTA. and the AE loss cannot pull the performance up from there.
        adapt_f1 = tf.nn.leaky_relu(adapt_f1)
        
        adapt_f2 = tf.layers.conv2d(inputs=adapt_f1,
                                    filters=64,
                                    kernel_size=1,
                                    padding='SAME',
                                    name='Axconv2',
                                    use_bias=True,
                                    activation=tf.identity,
                                    kernel_initializer = tf.initializers.random_normal(mean=1.0/64.0, stddev=tf.math.sqrt(2.0 / 64.0)))
        if instance_normalze == 1:
            adapt_f2 = layers.instance_normalize(adapt_f2)
        adapt_f2 = tf.nn.leaky_relu(adapt_f2)

        adapt_output = tf.layers.conv2d(inputs=adapt_f2,
                                        filters=1,
                                        kernel_size=1,
                                        padding='SAME',
                                        name='Axconv3',
                                        use_bias=True,
                                        activation=tf.identity,
                                        kernel_initializer = tf.initializers.random_normal(mean=1.0/64.0, stddev=tf.math.sqrt(2.0 / 64.0)))
        
    return adapt_output

# ======================================================================
# For TTA-AE (Yufan He, MedIA 2021)
# ======================================================================
# def unet2D_i2l_with_adaptors(images,
#                              nlabels,
#                              training_pl,
#                              return_features=False):

#     n0 = 16
#     n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0
    
#     # ====================================
#     # 1st Conv block - two conv layers, followed by max-pooling
#     # ====================================
#     with tf.variable_scope('i2l_mapper') as scope:
#         conv1_1 = layers.conv2D_layer_bn(x=images, name='conv1_1', num_filters=n1, training = training_pl)
#         conv1_2 = layers.conv2D_layer_bn(x=conv1_1, name='conv1_2', num_filters=n1, training = training_pl)
#         pool1 = layers.max_pool_layer2d(conv1_2)
#         # Feature adaptor 1 (These weights will be initialized to identity mappings in the TTA file. Can't figure out how to do this directly here..)
#         pool1_adapted = tf.layers.conv2d(inputs=pool1, filters=n1, kernel_size=1, padding='SAME', name='adaptAf_A1', use_bias=True, activation=tf.identity, kernel_initializer = tf.initializers.random_uniform(minval=1.0, maxval=1.0))

#         # ====================================
#         # 2nd Conv block
#         # ====================================
#         conv2_1 = layers.conv2D_layer_bn(x=pool1_adapted, name='conv2_1', num_filters=n2, training = training_pl)
#         conv2_2 = layers.conv2D_layer_bn(x=conv2_1, name='conv2_2', num_filters=n2, training = training_pl)
#         pool2 = layers.max_pool_layer2d(conv2_2)
#         # Feature adaptor 2 (These weights will be initialized to identity mappings in the TTA file. Can't figure out how to do this directly here..)
#         pool2_adapted = tf.layers.conv2d(inputs=pool2, filters=n2, kernel_size=1, padding='SAME', name='adaptAf_A2', use_bias=True, activation=tf.identity, kernel_initializer = tf.initializers.random_uniform(minval=1.0, maxval=1.0))

#         # ====================================
#         # 3rd Conv block
#         # ====================================
#         conv3_1 = layers.conv2D_layer_bn(x=pool2_adapted, name='conv3_1', num_filters=n3, training = training_pl)
#         conv3_2 = layers.conv2D_layer_bn(x=conv3_1, name='conv3_2', num_filters=n3, training = training_pl)
#         pool3 = layers.max_pool_layer2d(conv3_1)
#         # Feature adaptor 3 (These weights will be initialized to identity mappings in the TTA file. Can't figure out how to do this directly here..)
#         pool3_adapted = tf.layers.conv2d(inputs=pool3, filters=n3, kernel_size=1, padding='SAME', name='adaptAf_A3', use_bias=True, activation=tf.identity, kernel_initializer = tf.initializers.random_uniform(minval=1.0, maxval=1.0))
    
#         # ====================================
#         # 4th Conv block and decoder blocks
#         # ====================================
#         conv4_1 = layers.conv2D_layer_bn(x=pool3_adapted, name='conv4_1', num_filters=n4, training = training_pl)
#         conv4_2 = layers.conv2D_layer_bn(x=conv4_1, name='conv4_2', num_filters=n4, training = training_pl)
    
#         # ====================================
#         # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
#         # ====================================
#         deconv3 = layers.bilinear_upsample2D(conv4_2, size = (tf.shape(conv3_2)[1],tf.shape(conv3_2)[2]), name='upconv3')
#         concat3 = tf.concat([deconv3, conv3_2], axis=-1)        
#         conv5_1 = layers.conv2D_layer_bn(x=concat3, name='conv5_1', num_filters=n3, training = training_pl)
#         conv5_2 = layers.conv2D_layer_bn(x=conv5_1, name='conv5_2', num_filters=n3, training = training_pl)
    
#         # ====================================
#         # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
#         # ====================================
#         deconv2 = layers.bilinear_upsample2D(conv5_2, size = (tf.shape(conv2_2)[1],tf.shape(conv2_2)[2]), name='upconv2')
#         concat2 = tf.concat([deconv2, conv2_2], axis=-1)        
#         conv6_1 = layers.conv2D_layer_bn(x=concat2, name='conv6_1', num_filters=n2, training = training_pl)
#         conv6_2 = layers.conv2D_layer_bn(x=conv6_1, name='conv6_2', num_filters=n2, training = training_pl)
    
#         # ====================================
#         # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
#         # ====================================
#         deconv1 = layers.bilinear_upsample2D(conv6_2, size = (tf.shape(conv1_2)[1],tf.shape(conv1_2)[2]), name='upconv1')
#         concat1 = tf.concat([deconv1, conv1_2], axis=-1)        
#         conv7_1 = layers.conv2D_layer_bn(x=concat1, name='conv7_1', num_filters=n1, training = training_pl)
#         conv7_2 = layers.conv2D_layer_bn(x=conv7_1, name='conv7_2', num_filters=n1, training = training_pl)
    
#         # ====================================
#         # Final conv layer - without batch normalization or activation
#         # ====================================
#         pred_logits = layers.conv2D_layer(x=conv7_2, name='pred', num_filters=nlabels, kernel_size=1)

#         # ====================================
#         # convert the logits to segmentation probabilities
#         # ====================================
#         pred_seg_soft = tf.nn.softmax(pred_logits, name='pred_seg_soft')

#     if return_features == False:
#         return pred_logits
#     else:
#         return pred_logits, tf.concat([pool1_adapted, deconv2], axis=-1), tf.concat([pool2_adapted, deconv3], axis=-1), tf.concat([pool3_adapted, conv4_2], axis=-1)

# ======================================================================
# For TTA-AE (Yufan He, MedIA 2021)
# ======================================================================
def unet2D_i2l_with_adaptors_new(images,
                                 nlabels,
                                 training_pl,
                                 return_features=False):

    n0 = 16
    n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0
    
    # ====================================
    # 1st Conv block - two conv layers, followed by max-pooling
    # ====================================
    with tf.variable_scope('i2l_mapper') as scope:
        conv1_1 = layers.conv2D_layer_bn(x=images, name='conv1_1', num_filters=n1, training = training_pl)
        conv1_2 = layers.conv2D_layer_bn(x=conv1_1, name='conv1_2', num_filters=n1, training = training_pl)
        pool1 = layers.max_pool_layer2d(conv1_2)

        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = layers.conv2D_layer_bn(x=pool1, name='conv2_1', num_filters=n2, training = training_pl)
        conv2_2 = layers.conv2D_layer_bn(x=conv2_1, name='conv2_2', num_filters=n2, training = training_pl)
        # Feature adaptor 1 (These weights will be initialized to identity mappings in the TTA file. Can't figure out how to do this directly here..)
        conv2_2_adapted = tf.layers.conv2d(inputs=conv2_2, filters=n2, kernel_size=1, padding='SAME', name='adaptAf_A1', use_bias=True, activation=tf.identity, kernel_initializer = tf.initializers.random_uniform(minval=1.0, maxval=1.0))
        pool2 = layers.max_pool_layer2d(conv2_2_adapted)

        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = layers.conv2D_layer_bn(x=pool2, name='conv3_1', num_filters=n3, training = training_pl)
        conv3_2 = layers.conv2D_layer_bn(x=conv3_1, name='conv3_2', num_filters=n3, training = training_pl)
        # Feature adaptor 2 (These weights will be initialized to identity mappings in the TTA file. Can't figure out how to do this directly here..)
        conv3_2_adapted = tf.layers.conv2d(inputs=conv3_2, filters=n3, kernel_size=1, padding='SAME', name='adaptAf_A2', use_bias=True, activation=tf.identity, kernel_initializer = tf.initializers.random_uniform(minval=1.0, maxval=1.0))
        pool3 = layers.max_pool_layer2d(conv3_2_adapted)
    
        # ====================================
        # 4th Conv block and decoder blocks
        # ====================================
        conv4_1 = layers.conv2D_layer_bn(x=pool3, name='conv4_1', num_filters=n4, training = training_pl)
        # Feature adaptor 3 (These weights will be initialized to identity mappings in the TTA file. Can't figure out how to do this directly here..)
        conv4_1_adapted = tf.layers.conv2d(inputs=conv4_1, filters=n4, kernel_size=1, padding='SAME', name='adaptAf_A3', use_bias=True, activation=tf.identity, kernel_initializer = tf.initializers.random_uniform(minval=1.0, maxval=1.0))
        conv4_2 = layers.conv2D_layer_bn(x=conv4_1_adapted, name='conv4_2', num_filters=n4, training = training_pl)
    
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv3 = layers.bilinear_upsample2D(conv4_2, size = (tf.shape(conv3_2)[1],tf.shape(conv3_2)[2]), name='upconv3')
        concat3 = tf.concat([deconv3, conv3_2_adapted], axis=-1)        
        conv5_1 = layers.conv2D_layer_bn(x=concat3, name='conv5_1', num_filters=n3, training = training_pl)
        conv5_2 = layers.conv2D_layer_bn(x=conv5_1, name='conv5_2', num_filters=n3, training = training_pl)
    
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv2 = layers.bilinear_upsample2D(conv5_2, size = (tf.shape(conv2_2)[1],tf.shape(conv2_2)[2]), name='upconv2')
        concat2 = tf.concat([deconv2, conv2_2_adapted], axis=-1)        
        conv6_1 = layers.conv2D_layer_bn(x=concat2, name='conv6_1', num_filters=n2, training = training_pl)
        conv6_2 = layers.conv2D_layer_bn(x=conv6_1, name='conv6_2', num_filters=n2, training = training_pl)
    
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv1 = layers.bilinear_upsample2D(conv6_2, size = (tf.shape(conv1_2)[1],tf.shape(conv1_2)[2]), name='upconv1')
        concat1 = tf.concat([deconv1, conv1_2], axis=-1)        
        conv7_1 = layers.conv2D_layer_bn(x=concat1, name='conv7_1', num_filters=n1, training = training_pl)
        conv7_2 = layers.conv2D_layer_bn(x=conv7_1, name='conv7_2', num_filters=n1, training = training_pl)
    
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        pred_logits = layers.conv2D_layer(x=conv7_2, name='pred', num_filters=nlabels, kernel_size=1)

        # ====================================
        # convert the logits to segmentation probabilities
        # ====================================
        pred_seg_soft = tf.nn.softmax(pred_logits, name='pred_seg_soft')

    if return_features == False:
        return pred_logits
    else:
        return pred_logits, tf.concat([conv2_2_adapted, conv6_2], axis=-1), tf.concat([conv3_2_adapted, conv5_2], axis=-1), tf.concat([conv4_1_adapted, conv4_2], axis=-1)

# ======================================================================
# normalization network
# ======================================================================
def net2D_i2i(images, exp_config, training, scope_reuse=False):
        
    with tf.variable_scope('image_normalizer') as scope:
        
        if scope_reuse:
            scope.reuse_variables()       
                
        num_layers = exp_config.norm_num_hidden_layers
        n1 = exp_config.norm_num_filters_per_layer
        k = exp_config.norm_kernel_size
        
        out = images
        
        for l in range(num_layers):
            out = tf.layers.conv2d(inputs=out,
                                   filters=n1,
                                   kernel_size=k,
                                   padding='SAME',
                                   name='norm_conv1_'+str(l+1),
                                   use_bias=True,
                                   activation=None)
            
            if exp_config.norm_batch_norm is True:
                out = tf.layers.batch_normalization(inputs=out, name = 'norm_conv1_' + str(l+1) + '_bn', training = training)
            
            if exp_config.norm_activation is 'elu':
                out = tf.nn.elu(out)
                
            elif exp_config.norm_activation is 'relu':
                out = tf.nn.relu(out)
                
            elif exp_config.norm_activation is 'rbf':            
                # ==================
                # fixed scale
                # ==================
                # scale = 0.2
                # ==================
                # learnable scale - one scale per layer
                # ==================
                # scale = tf.Variable(initial_value = 0.2, name = 'scale_'+str(l+1))
                # ==================
                # learnable scale - one scale activation unit
                # ==================
                # init_value = tf.random_normal([1,1,1,n1], mean=0.2, stddev=0.05)
                # scale = tf.Variable(initial_value = init_value, name = 'scale_'+str(l+1))
                scale = tf.get_variable(name = 'scale_'+str(l+1), shape=[1,1,1,n1])
                out = tf.exp(-(out**2) / (scale**2))
        
        delta = tf.layers.conv2d(inputs=out,
                                  filters=1,
                                  kernel_size=k,
                                  padding='SAME',
                                  name='norm_conv1_'+str(num_layers+1),
                                  use_bias=True,
                                  activation=tf.identity)
        
        # =========================
        # Only model an additive residual effect with the normalizer
        # =========================
        output = images + delta
        
    return output, delta

# ======================================================================
# 2D Unet for mapping from images to segmentation labels
# ======================================================================
# def unet2D_i2l(images,
#                nlabels,
#                training_pl,
#                scope_reuse=False,
#                return_features=False): 

#     n0 = 16
#     n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0
    
#     with tf.variable_scope('i2l_mapper') as scope:
        
#         if scope_reuse:
#             scope.reuse_variables()
        
#         # ====================================
#         # 1st Conv block - two conv layers, followed by max-pooling
#         # ====================================
#         conv1_1 = layers.conv2D_layer_bn(x=images, name='conv1_1', num_filters=n1, training = training_pl)
#         conv1_2 = layers.conv2D_layer_bn(x=conv1_1, name='conv1_2', num_filters=n1, training = training_pl)
#         pool1 = layers.max_pool_layer2d(conv1_2)
    
#         # ====================================
#         # 2nd Conv block
#         # ====================================
#         conv2_1 = layers.conv2D_layer_bn(x=pool1, name='conv2_1', num_filters=n2, training = training_pl)
#         conv2_2 = layers.conv2D_layer_bn(x=conv2_1, name='conv2_2', num_filters=n2, training = training_pl)
#         pool2 = layers.max_pool_layer2d(conv2_2)
    
#         # ====================================
#         # 3rd Conv block
#         # ====================================
#         conv3_1 = layers.conv2D_layer_bn(x=pool2, name='conv3_1', num_filters=n3, training = training_pl)
#         conv3_2 = layers.conv2D_layer_bn(x=conv3_1, name='conv3_2', num_filters=n3, training = training_pl)
#         pool3 = layers.max_pool_layer2d(conv3_1)
#         # !!! pool3 should ideally have gotten as input conv3_2, instead it is getting conv3_1 !!!
    
#         # ====================================
#         # 4th Conv block
#         # ====================================
#         conv4_1 = layers.conv2D_layer_bn(x=pool3, name='conv4_1', num_filters=n4, training = training_pl)
#         conv4_2 = layers.conv2D_layer_bn(x=conv4_1, name='conv4_2', num_filters=n4, training = training_pl)
    
#         # ====================================
#         # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
#         # ====================================
#         deconv3 = layers.bilinear_upsample2D(conv4_2, size = (tf.shape(conv3_2)[1],tf.shape(conv3_2)[2]), name='upconv3')
#         concat3 = tf.concat([deconv3, conv3_2], axis=-1)        
#         conv5_1 = layers.conv2D_layer_bn(x=concat3, name='conv5_1', num_filters=n3, training = training_pl)
#         conv5_2 = layers.conv2D_layer_bn(x=conv5_1, name='conv5_2', num_filters=n3, training = training_pl)
    
#         # ====================================
#         # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
#         # ====================================
#         deconv2 = layers.bilinear_upsample2D(conv5_2, size = (tf.shape(conv2_2)[1],tf.shape(conv2_2)[2]), name='upconv2')
#         concat2 = tf.concat([deconv2, conv2_2], axis=-1)        
#         conv6_1 = layers.conv2D_layer_bn(x=concat2, name='conv6_1', num_filters=n2, training = training_pl)
#         conv6_2 = layers.conv2D_layer_bn(x=conv6_1, name='conv6_2', num_filters=n2, training = training_pl)
    
#         # ====================================
#         # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
#         # ====================================
#         deconv1 = layers.bilinear_upsample2D(conv6_2, size = (tf.shape(conv1_2)[1],tf.shape(conv1_2)[2]), name='upconv1')
#         concat1 = tf.concat([deconv1, conv1_2], axis=-1)        
#         conv7_1 = layers.conv2D_layer_bn(x=concat1, name='conv7_1', num_filters=n1, training = training_pl)
#         conv7_2 = layers.conv2D_layer_bn(x=conv7_1, name='conv7_2', num_filters=n1, training = training_pl)
    
#         # ====================================
#         # Final conv layer - without batch normalization or activation
#         # ====================================
#         pred_logits = layers.conv2D_layer(x=conv7_2, name='pred', num_filters=nlabels, kernel_size=1)

#         # ====================================
#         # convert the logits to segmentation probabilities
#         # ====================================
#         pred_seg_soft = tf.nn.softmax(pred_logits, name='pred_seg_soft')

#     if return_features == False:
#         return pred_logits
#     else:
#         return pred_logits, tf.concat([pool1, deconv2], axis=-1), tf.concat([pool2, deconv3], axis=-1), tf.concat([pool3, conv4_2], axis=-1)

# ======================================================================
# 2D Unet for mapping from images to segmentation labels
# Fixes two bugs as compared to the earlier unet2D_i2l:
# 1. conv3_2 goes into pool3, instead of conv3_1 as was happening before
# 2. features after conv are returned to be fed into AEs (for TTA-AE) instead of features after pool.
# This is more similar to what is done in He 2021.
# ======================================================================
def unet2D_i2l_new(images,
                   nlabels,
                   training_pl,
                   scope_reuse=False,
                   return_features=False): 

    n0 = 16
    n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0
    
    with tf.variable_scope('i2l_mapper') as scope:
        
        if scope_reuse:
            scope.reuse_variables()
        
        # ====================================
        # 1st Conv block - two conv layers, followed by max-pooling
        # ====================================
        conv1_1 = layers.conv2D_layer_bn(x=images, name='conv1_1', num_filters=n1, training = training_pl)
        conv1_2 = layers.conv2D_layer_bn(x=conv1_1, name='conv1_2', num_filters=n1, training = training_pl)
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = layers.conv2D_layer_bn(x=pool1, name='conv2_1', num_filters=n2, training = training_pl)
        conv2_2 = layers.conv2D_layer_bn(x=conv2_1, name='conv2_2', num_filters=n2, training = training_pl)
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = layers.conv2D_layer_bn(x=pool2, name='conv3_1', num_filters=n3, training = training_pl)
        conv3_2 = layers.conv2D_layer_bn(x=conv3_1, name='conv3_2', num_filters=n3, training = training_pl)
        pool3 = layers.max_pool_layer2d(conv3_2)
    
        # ====================================
        # 4th Conv block
        # ====================================
        conv4_1 = layers.conv2D_layer_bn(x=pool3, name='conv4_1', num_filters=n4, training = training_pl)
        conv4_2 = layers.conv2D_layer_bn(x=conv4_1, name='conv4_2', num_filters=n4, training = training_pl)
    
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv3 = layers.bilinear_upsample2D(conv4_2, size = (tf.shape(conv3_2)[1],tf.shape(conv3_2)[2]), name='upconv3')
        concat3 = tf.concat([deconv3, conv3_2], axis=-1)        
        conv5_1 = layers.conv2D_layer_bn(x=concat3, name='conv5_1', num_filters=n3, training = training_pl)
        conv5_2 = layers.conv2D_layer_bn(x=conv5_1, name='conv5_2', num_filters=n3, training = training_pl)
    
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv2 = layers.bilinear_upsample2D(conv5_2, size = (tf.shape(conv2_2)[1],tf.shape(conv2_2)[2]), name='upconv2')
        concat2 = tf.concat([deconv2, conv2_2], axis=-1)        
        conv6_1 = layers.conv2D_layer_bn(x=concat2, name='conv6_1', num_filters=n2, training = training_pl)
        conv6_2 = layers.conv2D_layer_bn(x=conv6_1, name='conv6_2', num_filters=n2, training = training_pl)
    
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        deconv1 = layers.bilinear_upsample2D(conv6_2, size = (tf.shape(conv1_2)[1],tf.shape(conv1_2)[2]), name='upconv1')
        concat1 = tf.concat([deconv1, conv1_2], axis=-1)        
        conv7_1 = layers.conv2D_layer_bn(x=concat1, name='conv7_1', num_filters=n1, training = training_pl)
        conv7_2 = layers.conv2D_layer_bn(x=conv7_1, name='conv7_2', num_filters=n1, training = training_pl)
    
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        pred_logits = layers.conv2D_layer(x=conv7_2, name='pred', num_filters=nlabels, kernel_size=1)

        # ====================================
        # convert the logits to segmentation probabilities
        # ====================================
        pred_seg_soft = tf.nn.softmax(pred_logits, name='pred_seg_soft')

    if return_features == False:
        return pred_logits
    else:
        return pred_logits, tf.concat([conv2_2, conv6_2], axis=-1), tf.concat([conv3_2, conv5_2], axis=-1), tf.concat([conv4_1, conv4_2], axis=-1)

# ======================================================================
# 2D autoencoder self supervised AUTOENCODER
# ======================================================================
def self_sup_autoencoder(inputs, training_pl, ae_features = 'xn'): 

    num_output_channels = inputs.shape[-1]
    n0 = 16
    n1, n2, n3, n4, n5 = 1*n0, 2*n0, 4*n0, 8*n0, 16*n0
    
    with tf.variable_scope('self_sup_ae_' + ae_features):
        
        # ====================================
        # 1st Conv block - two conv layers, followed by max-pooling
        # ====================================
        conv1_1 = layers.conv2D_layer_bn(x=inputs, name='conv1_1', num_filters=n1, training=training_pl)
        conv1_2 = layers.conv2D_layer_bn(x=conv1_1, name='conv1_2', num_filters=n1, training=training_pl)
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = layers.conv2D_layer_bn(x=pool1, name='conv2_1', num_filters=n2, training=training_pl)
        conv2_2 = layers.conv2D_layer_bn(x=conv2_1, name='conv2_2', num_filters=n2, training=training_pl)
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = layers.conv2D_layer_bn(x=pool2, name='conv3_1', num_filters=n3, training=training_pl)
        conv3_2 = layers.conv2D_layer_bn(x=conv3_1, name='conv3_2', num_filters=n3, training=training_pl)
        pool3 = layers.max_pool_layer2d(conv3_2)
    
        # ====================================
        # 4th Conv block
        # ====================================
        conv4_1 = layers.conv2D_layer_bn(x=pool3, name='conv4_1', num_filters=n4, training=training_pl)
        conv4_2 = layers.conv2D_layer_bn(x=conv4_1, name='conv4_2', num_filters=n4, training=training_pl)
        pool4 = layers.max_pool_layer2d(conv4_2)
        
        # ====================================
        # 5th Conv block
        # ====================================
        conv5_1 = layers.conv2D_layer_bn(x=pool4, name='conv5_1', num_filters=n5, training=training_pl)
        conv5_2 = layers.conv2D_layer_bn(x=conv5_1, name='conv5_2', num_filters=n5, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, followed by 2 conv layers
        # ====================================
        upsample1 = layers.bilinear_upsample2D(conv5_2, size = (tf.shape(conv4_2)[1], tf.shape(conv4_2)[2]), name='upsample1')
        conv6_1 = layers.conv2D_layer_bn(x=upsample1, name='conv6_1', num_filters=n4, training=training_pl)
        conv6_2 = layers.conv2D_layer_bn(x=conv6_1, name='conv6_2', num_filters=n4, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, followed by 2 conv layers
        # ====================================
        upsample2 = layers.bilinear_upsample2D(conv6_2, size = (tf.shape(conv3_2)[1], tf.shape(conv3_2)[2]), name='upsample2')    
        conv7_1 = layers.conv2D_layer_bn(x=upsample2, name='conv7_1', num_filters=n3, training=training_pl)
        conv7_2 = layers.conv2D_layer_bn(x=conv7_1, name='conv7_2', num_filters=n3, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, followed by 2 conv layers
        # ====================================
        upsample3 = layers.bilinear_upsample2D(conv7_2, size = (tf.shape(conv2_2)[1], tf.shape(conv2_2)[2]), name='upsample3')    
        conv8_1 = layers.conv2D_layer_bn(x=upsample3, name='conv8_1', num_filters=n2, training=training_pl)
        conv8_2 = layers.conv2D_layer_bn(x=conv8_1, name='conv8_2', num_filters=n2, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, followed by 2 conv layers
        # ====================================
        upsample4 = layers.bilinear_upsample2D(conv8_2, size = (tf.shape(conv1_2)[1], tf.shape(conv1_2)[2]), name='upsample4')  
        conv9_1 = layers.conv2D_layer_bn(x=upsample4, name='conv9_1', num_filters=n1, training=training_pl)
        conv9_2 = layers.conv2D_layer_bn(x=conv9_1, name='conv9_2', num_filters=n1, training=training_pl)
            
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        outputs = layers.conv2D_layer(x=conv9_2, name='output_layer', num_filters=num_output_channels, kernel_size=1)

    return outputs

# ======================================================================
# 2D autoencoder self supervised AUTOENCODER
# ======================================================================
def self_sup_autoencoder_like_yufan(inputs, training_pl, ae_features = 'xn'): 

    num_output_channels = inputs.shape[-1]

    # They use lower number of channels as the spatial dimensionality increases.
    # Not sure what is the idea behind this, but following the same architecture.
    if ae_features in ['xn', 'y']:
        n_channels = [32, 16, 8]
    elif ae_features in ['f1', 'f2', 'f3']:
        n_channels = [64, 32, 16]
    
    with tf.variable_scope('self_sup_ae_' + ae_features):
        
        # ====================================
        # 1st Conv layer, followed by instance normalization, followed by max-pooling
        # ====================================
        conv1 = layers.conv2D_layer_in(x=inputs, name='conv1', num_filters=n_channels[0], training=training_pl)
        pool1 = layers.max_pool_layer2d(conv1)
    
        # ====================================
        # 2nd Conv layer, followed by instance normalization, followed by max-pooling
        # ====================================
        conv2 = layers.conv2D_layer_in(x=pool1, name='conv2', num_filters=n_channels[1], training=training_pl)
        pool2 = layers.max_pool_layer2d(conv2)
    
        # ====================================
        # 3rd Conv layer, followed by instance normalization
        # ====================================
        conv3 = layers.conv2D_layer_in(x=pool2, name='conv3', num_filters=n_channels[2], training=training_pl)
            
        # ====================================
        # Upsampling via bilinear upsampling, followed by 1 conv layer, followed by instance normalization
        # ====================================
        upsample1 = layers.bilinear_upsample2D(conv3, size = (tf.shape(conv2)[1], tf.shape(conv2)[2]), name='upsample1')
        conv4 = layers.conv2D_layer_in(x=upsample1, name='conv4', num_filters=n_channels[1], training=training_pl)

        # ====================================
        # Upsampling via bilinear upsampling, followed by 1 conv layer, followed by instance normalization
        # ====================================
        upsample2 = layers.bilinear_upsample2D(conv4, size = (tf.shape(conv1)[1], tf.shape(conv1)[2]), name='upsample2')
        conv5 = layers.conv2D_layer_in(x=upsample2, name='conv5', num_filters=n_channels[0], training=training_pl)
                    
        # ====================================
        # Final conv layer - without normalization or activation
        # ====================================
        outputs = layers.conv2D_layer(x=conv5, name='output_layer', num_filters=num_output_channels, kernel_size=1)

    return outputs

# ======================================================================
# 2D autoencoder self supervised VARIATIONAL AUTOENCODER
# ======================================================================
def self_sup_variational_autoencoder(inputs, training_pl): 

    n0 = 16
    n1, n2, n3, n4, n5 = 1*n0, 2*n0, 4*n0, 8*n0, 16*n0
    
    with tf.variable_scope('self_sup_vae'):
        
        # ====================================
        # 1st Conv block - two conv layers, followed by max-pooling
        # ====================================
        conv1_1 = layers.conv2D_layer_bn(x=inputs, name='conv1_1', num_filters=n1, training=training_pl)
        conv1_2 = layers.conv2D_layer_bn(x=conv1_1, name='conv1_2', num_filters=n1, training=training_pl)
        pool1 = layers.max_pool_layer2d(conv1_2)
    
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = layers.conv2D_layer_bn(x=pool1, name='conv2_1', num_filters=n2, training=training_pl)
        conv2_2 = layers.conv2D_layer_bn(x=conv2_1, name='conv2_2', num_filters=n2, training=training_pl)
        pool2 = layers.max_pool_layer2d(conv2_2)
    
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = layers.conv2D_layer_bn(x=pool2, name='conv3_1', num_filters=n3, training=training_pl)
        conv3_2 = layers.conv2D_layer_bn(x=conv3_1, name='conv3_2', num_filters=n3, training=training_pl)
        pool3 = layers.max_pool_layer2d(conv3_2)
    
        # ====================================
        # 4th Conv block
        # ====================================
        conv4_1 = layers.conv2D_layer_bn(x=pool3, name='conv4_1', num_filters=n4, training=training_pl)
        conv4_2 = layers.conv2D_layer_bn(x=conv4_1, name='conv4_2', num_filters=n4, training=training_pl)
        pool4 = layers.max_pool_layer2d(conv4_2)
        
        # ====================================
        # 5th Conv block
        # ====================================
        conv5_1 = layers.conv2D_layer_bn(x=pool4, name='conv5_1', num_filters=n5, training=training_pl)
        conv5_2 = layers.conv2D_layer_bn(x=conv5_1, name='conv5_2', num_filters=n5, training=training_pl)

        # ====================================
        # compute mean and var of the latent representation
        # ====================================
        z_mu = layers.conv2D_layer(x=conv5_2, name='z_mu', num_filters=n5, kernel_size=1)
        z_std = layers.conv2D_layer(x=conv5_2, name='z_std', num_filters=n5, kernel_size=1)
        z_rand = tf.random_normal(tf.shape(z_mu), 0., 1., dtype=tf.float32)
        z = z_mu + z_std * z_rand
        
        # ====================================
        # Upsampling via bilinear upsampling, followed by 2 conv layers
        # ====================================
        upsample1 = layers.bilinear_upsample2D(z, size = (tf.shape(conv4_2)[1], tf.shape(conv4_2)[2]), name='upsample1')
        conv6_1 = layers.conv2D_layer_bn(x=upsample1, name='conv6_1', num_filters=n4, training=training_pl)
        conv6_2 = layers.conv2D_layer_bn(x=conv6_1, name='conv6_2', num_filters=n4, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, followed by 2 conv layers
        # ====================================
        upsample2 = layers.bilinear_upsample2D(conv6_2, size = (tf.shape(conv3_2)[1], tf.shape(conv3_2)[2]), name='upsample2')    
        conv7_1 = layers.conv2D_layer_bn(x=upsample2, name='conv7_1', num_filters=n3, training=training_pl)
        conv7_2 = layers.conv2D_layer_bn(x=conv7_1, name='conv7_2', num_filters=n3, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, followed by 2 conv layers
        # ====================================
        upsample3 = layers.bilinear_upsample2D(conv7_2, size = (tf.shape(conv2_2)[1], tf.shape(conv2_2)[2]), name='upsample3')    
        conv8_1 = layers.conv2D_layer_bn(x=upsample3, name='conv8_1', num_filters=n2, training=training_pl)
        conv8_2 = layers.conv2D_layer_bn(x=conv8_1, name='conv8_2', num_filters=n2, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, followed by 2 conv layers
        # ====================================
        upsample4 = layers.bilinear_upsample2D(conv8_2, size = (tf.shape(conv1_2)[1], tf.shape(conv1_2)[2]), name='upsample4')  
        conv9_1 = layers.conv2D_layer_bn(x=upsample4, name='conv9_1', num_filters=n1, training=training_pl)
        conv9_2 = layers.conv2D_layer_bn(x=conv9_1, name='conv9_2', num_filters=n1, training=training_pl)
            
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        outputs = layers.conv2D_layer(x=conv9_2, name='output_layer', num_filters=1, kernel_size=1)

    return outputs, z_mu, z_std

# ======================================================================
# 3D Unet for label autoencoder
# ======================================================================
def self_sup_denoising_autoencoder_3D(inputs, nlabels, training_pl): 

    n0 = 16
    n1, n2, n3, n4 = 1*n0, 2*n0, 4*n0, 8*n0
    
    with tf.variable_scope('l2l_mapper'):
        
        # ====================================
        # 1st Conv block - two conv layers, followed by max-pooling
        # ====================================
        conv1_1 = layers.conv3D_layer_bn(x=inputs, name='conv1_1', num_filters=n1, training=training_pl)
        conv1_2 = layers.conv3D_layer_bn(x=conv1_1, name='conv1_2', num_filters=n1, training=training_pl)
        pool1 = layers.max_pool_layer3d(conv1_2)
    
        # ====================================
        # 2nd Conv block
        # ====================================
        conv2_1 = layers.conv3D_layer_bn(x=pool1, name='conv2_1', num_filters=n2, training=training_pl)
        conv2_2 = layers.conv3D_layer_bn(x=conv2_1, name='conv2_2', num_filters=n2, training=training_pl)
        pool2 = layers.max_pool_layer3d(conv2_2)
    
        # ====================================
        # 3rd Conv block
        # ====================================
        conv3_1 = layers.conv3D_layer_bn(x=pool2, name='conv3_1', num_filters=n3, training=training_pl)
        conv3_2 = layers.conv3D_layer_bn(x=conv3_1, name='conv3_2', num_filters=n3, training=training_pl)
        pool3 = layers.max_pool_layer3d(conv3_2)
                
        # ====================================
        # 4th Conv block
        # ====================================
        conv5_1 = layers.conv3D_layer_bn(x=pool3, name='conv5_1', num_filters=n4, training=training_pl)
        conv5_2 = layers.conv3D_layer_bn(x=conv5_1, name='conv5_2', num_filters=n4, training=training_pl)
                
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample2 = layers.bilinear_upsample3D(conv5_2, factor = 2, name='upsample2')
        concat2 = tf.concat([upsample2, conv3_2], axis=-1)
        conv7_1 = layers.conv3D_layer_bn(x=concat2, name='conv7_1', num_filters=n3, training=training_pl)
        conv7_2 = layers.conv3D_layer_bn(x=conv7_1, name='conv7_2', num_filters=n3, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, concatenation (skip connection), followed by 2 conv layers
        # ====================================
        upsample3 = layers.bilinear_upsample3D(conv7_2, factor = 2, name='upsample3')  
        concat3 = tf.concat([upsample3, conv2_2], axis=-1)
        conv8_1 = layers.conv3D_layer_bn(x=concat3, name='conv8_1', num_filters=n2, training=training_pl)
        conv8_2 = layers.conv3D_layer_bn(x=conv8_1, name='conv8_2', num_filters=n2, training=training_pl)
        
        # ====================================
        # Upsampling via bilinear upsampling, followed by 2 conv layers
        # NO SKIP CONNECTION IN THIS LAYER
        # ====================================
        upsample4 = layers.bilinear_upsample3D(conv8_2, factor = 2, name='upsample4')   
        conv9_1 = layers.conv3D_layer_bn(x=upsample4, name='conv9_1', num_filters=n1, training=training_pl)
        # conv9_2 = layers.conv3D_layer_bn(x=conv9_1, name='conv9_2', num_filters=n1, training=training_pl)
        outputs = layers.conv3D_layer_bn(x=conv9_1, name='conv9_2', num_filters=nlabels, training=training_pl, activation=None, bn=False)
            
        # ====================================
        # Final conv layer - without batch normalization or activation
        # ====================================
        # outputs = layers.conv3D_layer(x=conv9_2, name='output_layer', num_filters=nlabels, kernel_size=1)
        # outputs = layers.conv3D_layer_bn(x=conv9_2, name='output_layer', training=training_pl, kernel_size=1, num_filters=nlabels, activation=None, padding="SAME", kernel_initializer=None, bn=False)
        
    return outputs