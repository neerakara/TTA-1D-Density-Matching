# ===============================================================
# visualization functions
# ===============================================================
import matplotlib
matplotlib.rcParams['xtick.labelsize'] = 20
matplotlib.rcParams['ytick.labelsize'] = 20
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import numpy as np
import logging
from scipy.stats import norm
from skimage import color
from skimage import io
import tensorflow as tf

# ==========================================================
# ==========================================================
def add_1_pixel_each_class(arr, nlabels=15):
    
    arr_ = np.copy(arr)
    for j in range(nlabels):
        arr_[0,j]=j
    
    return arr_

# ==========================================================
# ==========================================================
def save_single_image(image,
                      savepath,
                      nlabels=3,
                      add_pixel_each_label=False,
                      cmap='gray',
                      colorbar=False,
                      climits = [],
                      dpi = 100):
        
    plt.figure(figsize=[20,20])            
    
    if add_pixel_each_label:
        image = add_1_pixel_each_class(image, nlabels)
                
    plt.imshow(image, cmap=cmap)
    if climits != []:
        plt.clim([climits[0], climits[1]])
    plt.axis('off')
    if colorbar:
        plt.colorbar()
    plt.savefig(savepath, bbox_inches='tight', dpi=dpi)
    plt.close()

# ==========================================================
# ==========================================================
def save_samples_downsampled(y,
                             savepath,
                             add_pixel_each_label=True,
                             cmap='tab20'):
        
    plt.figure(figsize=[20,10])
    
    for i in range(4):
    
        for j in range(8):
        
            plt.subplot(4, 8, 8*i+j+1)
            
            if add_pixel_each_label:
                labels_this_slice = add_1_pixel_each_class(y[8*i+j,:,:])
            else:
                labels_this_slice = y[8*i+j,:,:]
                
            plt.imshow(labels_this_slice, cmap=cmap)
            plt.colorbar()

    plt.savefig(savepath, bbox_inches='tight')
    plt.close()
    
# ==========================================================
# ==========================================================
def save_samples_downsampled2(y,
                              savepath,
                              add_pixel_each_label=True,
                              cmap='tab20',
                              colorbar=False,
                              climits = []):
        
    plt.figure(figsize=[20,10])
    
    for i in range(4):
    
        for j in range(8):
        
            plt.subplot(4, 8, 8*i+j+1)
            
            if add_pixel_each_label:
                labels_this_slice = add_1_pixel_each_class(y[8*i+j,:,:])
            else:
                labels_this_slice = y[8*i+j,...]
                
            plt.imshow(labels_this_slice, cmap=cmap)
            plt.colorbar()

    plt.savefig(savepath, bbox_inches='tight')
    plt.close()
    
# ==========================================================
# ==========================================================       
def save_sample_prediction_results(x,
                                   x_norm,
                                   y_pred,
                                   gt,
                                   num_rotations,
                                   savepath,
                                   nlabels,
                                   ids):

    nc = len(ids)
    nr = 5

    y_pred_ = np.copy(y_pred)
    gt_ = np.copy(gt)
    
    # add one pixel of each class to get consistent colors for each class in the visualization
    for i in range(nlabels):
        for idx in ids:
            y_pred_[0,i,idx] = i
            gt_[0,i,idx] = i
            
    # make a binary mask showing locations of incorrect predictions
    incorrect_mask = np.zeros_like(gt_)
    incorrect_mask[np.where(gt_ != y_pred_)] = 1
        
    plt.figure(figsize=[3*nc, 3*nr])
    
    for c in range(nc): 
        
        x_vis = np.rot90(x[:, :, ids[c]], k=num_rotations)
        x_norm_vis = np.rot90(x_norm[:, :, ids[c]], k=num_rotations)
        y_pred_vis = np.rot90(y_pred_[:, :, ids[c]], k=num_rotations)
        gt_vis = np.rot90(gt_[:, :, ids[c]], k=num_rotations)
        incorrect_mask_vis = np.rot90(incorrect_mask[:, :, ids[c]], k=num_rotations)

        plt.subplot(nr, nc, nc*0 + c + 1); plt.imshow(x_vis, cmap='gray'); plt.colorbar(); plt.title('Image')
        plt.subplot(nr, nc, nc*1 + c + 1); plt.imshow(x_norm_vis, cmap='gray'); plt.colorbar(); plt.title('Normalized')
        plt.subplot(nr, nc, nc*2 + c + 1); plt.imshow(y_pred_vis, cmap='tab20'); plt.colorbar(); plt.title('Prediction')
        plt.subplot(nr, nc, nc*3 + c + 1); plt.imshow(gt_vis, cmap='tab20'); plt.colorbar(); plt.title('Ground Truth')
        plt.subplot(nr, nc, nc*4 + c + 1); plt.imshow(incorrect_mask_vis, cmap='tab20'); plt.colorbar(); plt.title('Incorrect pixels')
        
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()
    
# ==========================================================
# ==========================================================       
def save_sample_results(x,
                        x_norm,
                        x_recon,
                        x_diff,
                        y,
                        gt,
                        savepath):
    
    ids = np.arange(0, x.shape[0], x.shape[0] // 8)
    
    y_ = np.copy(y)
    gt_ = np.copy(gt)
    
    # add one pixel of each class to get consistent colors for each class in the visualization
    for i in range(15):
        for idx in ids:
            y_[idx,0,i] = i
            gt_[idx,0,i] = i
    
    nc = 6
    plt.figure(figsize=[nc*3, 3*len(ids)])
    for i in range(len(ids)): 
        plt.subplot(len(ids), nc, nc*i+1); plt.imshow(x[ids[i],:,:], cmap='gray'); plt.clim([0,1.1]); plt.colorbar(); plt.title('test image')
        plt.subplot(len(ids), nc, nc*i+2); plt.imshow(x_norm[ids[i],:,:], cmap='gray'); plt.colorbar(); plt.title('normalized image')
        plt.subplot(len(ids), nc, nc*i+3); plt.imshow(x_recon[ids[i],:,:], cmap='gray'); plt.colorbar(); plt.title('reconstructed image')
        plt.subplot(len(ids), nc, nc*i+4); plt.imshow(x_diff[ids[i],:,:], cmap='gray'); plt.colorbar(); plt.title('AE error image')
        plt.subplot(len(ids), nc, nc*i+5); plt.imshow(y_[ids[i],:,:], cmap='tab20'); plt.colorbar(); plt.title('pred')
        plt.subplot(len(ids), nc, nc*i+6); plt.imshow(gt_[ids[i],:,:], cmap='tab20'); plt.colorbar(); plt.title('ground truth')
    
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()
    
# ==========================================================
# ==========================================================       
def save_sample_results_vae_dae(x,
                                x_norm,
                                diff_x_norm__x_norm_initial,
                                x_norm_recon,
                                diff_x_norm__x_norm_recon,
                                y_pred,
                                y_pred_dae,
                                gt,
                                savepath):
    
    ids = np.arange(0, x.shape[0], x.shape[0] // 8)
    
    y_pred_ = np.copy(y_pred)
    y_pred_dae_ = np.copy(y_pred_dae)
    gt_ = np.copy(gt)
    
    # add one pixel of each class to get consistent colors for each class in the visualization
    for i in range(15):
        for idx in ids:
            y_pred_[idx,0,i] = i
            y_pred_dae_[idx,0,i] = i
            gt_[idx,0,i] = i
    
    nc = 8
    plt.figure(figsize=[nc*3, 3*len(ids)])
    for i in range(len(ids)): 
        plt.subplot(len(ids), nc, nc*i+1); plt.imshow(x[ids[i],:,:], cmap='gray'); plt.clim([0,1.1]); plt.colorbar(); plt.title('test image')
        plt.subplot(len(ids), nc, nc*i+2); plt.imshow(x_norm[ids[i],:,:], cmap='gray'); plt.colorbar(); plt.title('x norm')
        plt.subplot(len(ids), nc, nc*i+3); plt.imshow(diff_x_norm__x_norm_initial[ids[i],:,:], cmap='gray'); plt.colorbar(); plt.title('x norm change')
        plt.subplot(len(ids), nc, nc*i+4); plt.imshow(x_norm_recon[ids[i],:,:], cmap='gray'); plt.colorbar(); plt.title('x norm recon')
        plt.subplot(len(ids), nc, nc*i+5); plt.imshow(diff_x_norm__x_norm_recon[ids[i],:,:], cmap='gray'); plt.colorbar(); plt.title('VAE error image')
        plt.subplot(len(ids), nc, nc*i+6); plt.imshow(y_pred_[ids[i],:,:], cmap='tab20'); plt.colorbar(); plt.title('pred')
        plt.subplot(len(ids), nc, nc*i+7); plt.imshow(y_pred_dae_[ids[i],:,:], cmap='tab20'); plt.colorbar(); plt.title('dae output')
        plt.subplot(len(ids), nc, nc*i+8); plt.imshow(gt_[ids[i],:,:], cmap='tab20'); plt.colorbar(); plt.title('ground truth')
    
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()
    
def plot_graph(a,
               b,
               save_path,
               x_range=None,
               y_range=None):
    plt.figure()
    plt.plot(a, b)
    if x_range != None:
        plt.xlim(x_range)
    if y_range != None:
        plt.ylim(y_range)
    plt.savefig(save_path)
    plt.close()

# ================================================================
# ================================================================
def write_feature_summaries(step,
                            summary_writer,
                            sess,
                            summary_features_sd,
                            display_sd_pl,
                            features_sd,
                            summary_features_td,
                            display_td_pl,
                            features_td):

    summary_writer.add_summary(sess.run(summary_features_sd, feed_dict = {display_sd_pl: prepare_for_tensorboard(stitch_features(features_sd))}), step)
    summary_writer.add_summary(sess.run(summary_features_td, feed_dict = {display_td_pl: prepare_for_tensorboard(stitch_features(features_td))}), step)

# ================================================================
# function to stitch all features of a particular iteration together
# ================================================================
def stitch_features(f): # shape of f --> [b_size, nx, ny, n_channels]
        
    nb, nx, ny, nc = f.shape
    stitched_feat = np.zeros((nx, nc*ny), dtype = np.float32)

    for c in range(nc):
        sy = c
        stitched_feat[:, sy*ny:(sy+1)*ny] = f[nb//2, :, :, c]
    
    return stitched_feat

# ================================================================
# ================================================================
def write_image_summaries(step,
                          summary_writer,
                          sess,
                          summary_images,
                          display_pl,
                          x,
                          xn,
                          y,
                          y_gt,
                          dx = 0,
                          dy = 0):
    
    # stitch images
    stitched_image = stitch_images(x, xn, y, y_gt, dx, dy)
    
    # make shape and type like tensorboard wants
    final_image = prepare_for_tensorboard(stitched_image)
    
    # write to tensorboard
    summary_writer.add_summary(sess.run(summary_images, feed_dict = {display_pl: final_image}), step)

# ================================================================
# ================================================================
def normalize_and_cast_to_uint8(x):

    x = x - np.min(x)
    x = x / np.max(x)
    x = x * 255
    x = x.astype(np.uint8)
    return x

# ================================================================
# function to stitch all images of a particular iteration together
# ================================================================
def stitch_images(x, xn, y, y_gt, dx, dy):
        
    nx, ny = x.shape[1:]
    nx = nx - 2*dx
    ny = ny - 2*dy
    stitched_image = np.zeros((4*nx, 5*ny), dtype = np.float32)
    vis_ids = np.linspace(0, x.shape[0], 9, dtype=np.uint8)[2:7]

    for i in range(5):
        sx = 0; sy = i; stitched_image[sx*nx:(sx+1)*nx, sy*ny:(sy+1)*ny] = x[vis_ids[i], dx:x.shape[1]-dx, dy:x.shape[2]-dy]
        sx = 1; sy = i; stitched_image[sx*nx:(sx+1)*nx, sy*ny:(sy+1)*ny] = xn[vis_ids[i], dx:xn.shape[1]-dx, dy:xn.shape[2]-dy]
        sx = 2; sy = i; stitched_image[sx*nx:(sx+1)*nx, sy*ny:(sy+1)*ny] = y[vis_ids[i], dx:y.shape[1]-dx, dy:y.shape[2]-dy]
        sx = 3; sy = i; stitched_image[sx*nx:(sx+1)*nx, sy*ny:(sy+1)*ny] = y_gt[vis_ids[i], dx:y_gt.shape[1]-dx, dy:y_gt.shape[2]-dy]
    
    return stitched_image

# ================================================================
# ================================================================
def prepare_for_tensorboard(x):

    x = normalize_and_cast_to_uint8(x)
    
    return np.expand_dims(np.expand_dims(x, axis=0), axis=-1)

# ================================================================
# ================================================================
def write_gaussians(step,
                    summary_writer,
                    sess,
                    summary_histograms,
                    display_pl,
                    sd_mu,
                    sd_var,
                    td_mu,
                    td_var,
                    savedir,
                    nlabels,
                    deltas = [0, 32, 96, 224, 480, 608, 672, 704],
                    num_channels = 3):

    if nlabels < num_channels:
        num_channels = nlabels
        
    # stitch images
    stitched_image = stitch_gaussians(sd_mu,
                                      sd_var,
                                      td_mu,
                                      td_var,
                                      savedir,
                                      nlabels,
                                      deltas,
                                      num_channels)
    
    # make shape and type like tensorboard wants
    final_image = prepare_for_tensorboard(stitched_image)
    
    # write to tensorboard
    summary_writer.add_summary(sess.run(summary_histograms, feed_dict = {display_pl: final_image}), step)

# ================================================================
# function to stitch all images of a particular iteration together
# ================================================================
def stitch_gaussians(sd_means,
                     sd_variances,
                     td_means,
                     td_variances,
                     savedir,
                     nlabels,
                     deltas,
                     num_channels):
        
    nx = 150
    ny = 150
    stitched_image = np.zeros((num_channels*nx, len(deltas)*ny), dtype = np.float32)

    sy = 0
    for delta in deltas:                
        for c in range(num_channels):
            sx = c
            stitched_image[sx*nx:(sx+1)*nx, sy*ny:(sy+1)*ny] = save_tmp_and_load(sd_means,
                                                                                 sd_variances,
                                                                                 td_means,
                                                                                 td_variances,
                                                                                 delta+c,
                                                                                 savedir)
        sy = sy+1

    return stitched_image

# ================================================================
# ================================================================
def save_tmp_and_load(sd_means,
                      sd_variances,
                      td_means,
                      td_variances,
                      c,
                      savedir):
    
    plt.figure(figsize=[1.5,1.5])
    x = np.linspace(sd_means[c] - 3*np.sqrt(sd_variances[c]), sd_means[c] + 3*np.sqrt(sd_variances[c]), 20)
    plt.plot(x, norm.pdf(x, sd_means[c], np.sqrt(sd_variances[c])), 'b')
    plt.plot(x, norm.pdf(x, td_means[c], np.sqrt(td_variances[c])), 'c')
    plt.savefig(savedir + '/tmp.png')
    plt.close()

    return io.imread(savedir + '/tmp.png')[:,:,0]

# ================================================================
# ================================================================
def save_1d_pdfs(gauss_params,
                 kde,
                 xaxis_range,
                 savepath):

    plt.figure(figsize=[5,5])
    x = np.arange(xaxis_range[0], xaxis_range[1] + xaxis_range[2], xaxis_range[2])
    plt.plot(x, norm.pdf(x, gauss_params[0], np.sqrt(gauss_params[1])), 'r')
    plt.plot(x, kde / (np.sum(kde)*xaxis_range[2]), 'b') # normalize KDE before plotting
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()

# ================================================================
# ================================================================
def save_1d_pdfs_pca(kde,
                     xaxis_range,
                     savepath):

    plt.figure(figsize=[5,5])
    x = np.arange(xaxis_range[0], xaxis_range[1] + xaxis_range[2], xaxis_range[2])
    plt.plot(x, kde / (np.sum(kde)*xaxis_range[2]), 'b') # normalize KDE before plotting
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()

# ================================================================
# ================================================================
def write_pdfs(step,
               summary_writer,
               sess,
               pdfs_summary,
               display_pl,
               pdfs_SD_g1_mu, pdfs_SD_g1_std, pdfs_TD_g1, x_g1,
               pdfs_SD_g2_mu, pdfs_SD_g2_std, pdfs_TD_g2, x_g2,
               pdfs_SD_g3_mu, pdfs_SD_g3_std, pdfs_TD_g3, x_g3,
               pdfs_SD_pca_mu, pdfs_SD_pca_std, pdfs_TD_pca, z,
               savedir):

    # stitch images
    stitched_image = stitch_pdfs(pdfs_SD_g1_mu, pdfs_SD_g1_std, pdfs_TD_g1, x_g1,
                                 pdfs_SD_g2_mu, pdfs_SD_g2_std, pdfs_TD_g2, x_g2,
                                 pdfs_SD_g3_mu, pdfs_SD_g3_std, pdfs_TD_g3, x_g3,
                                 pdfs_SD_pca_mu, pdfs_SD_pca_std, pdfs_TD_pca, z,
                                 savedir)
    
    # make shape and type like tensorboard wants
    final_image = prepare_for_tensorboard(stitched_image)
    
    # write to tensorboard
    summary_writer.add_summary(sess.run(pdfs_summary, feed_dict = {display_pl: final_image}), step)

# ================================================================
# function to stitch all images of a particular iteration together
# ================================================================
def stitch_pdfs(pdfs_SD_g1_mu, pdfs_SD_g1_std, pdfs_TD_g1, x_g1,
                pdfs_SD_g2_mu, pdfs_SD_g2_std, pdfs_TD_g2, x_g2,
                pdfs_SD_g3_mu, pdfs_SD_g3_std, pdfs_TD_g3, x_g3,
                pdfs_SD_pca_mu, pdfs_SD_pca_std, pdfs_TD_pca, z,
                savedir):
        
    nx = 150
    ny = 150
    num_layers_to_visualize = 9 # 1_1, 2_1, 3_1, 5_1, 6_1, 7_1, 7_2, logits, pca
    num_channels_per_layer = 5

    # show first 5 channels for all the 7 layers (c1_1, c2_1, c3_1, c4_1, c5_1, c6_1, c7_1)
    stitched_image = np.zeros((num_channels_per_layer*nx, num_layers_to_visualize*ny), dtype = np.float32)

    # visualize group 1 channels (1_1, 2_1, 3_1, 5_1, 6_1, 7_1)
    sy = 0
    for delta in [0, 32, 96, 480, 608, 672]:
        for c in range(num_channels_per_layer):
            sx = c
            stitched_image[sx*nx:(sx+1)*nx, sy*ny:(sy+1)*ny] = save_tmp_pdf_and_load(pdfs_SD_g1_mu,
                                                                                     pdfs_SD_g1_std,
                                                                                     pdfs_TD_g1,
                                                                                     x_g1,
                                                                                     delta+c,
                                                                                     savedir)
        sy = sy+1

    # visualize group 2 channels (7_2)
    for c in range(num_channels_per_layer):
        sx = c
        stitched_image[sx*nx:(sx+1)*nx, sy*ny:(sy+1)*ny] = save_tmp_pdf_and_load(pdfs_SD_g2_mu,
                                                                                 pdfs_SD_g2_std,
                                                                                 pdfs_TD_g2,
                                                                                 x_g2,
                                                                                 c,
                                                                                 savedir)
    sy = sy+1

    # visualize group 3 channels (softmax)
    for c in range(num_channels_per_layer):
        if c < pdfs_TD_g3.shape[0]:
            sx = c
            stitched_image[sx*nx:(sx+1)*nx, sy*ny:(sy+1)*ny] = save_tmp_pdf_and_load(pdfs_SD_g3_mu,
                                                                                     pdfs_SD_g3_std,
                                                                                     pdfs_TD_g3,
                                                                                     x_g3,
                                                                                     c,
                                                                                     savedir)
    sy = sy+1

    # visualize pca 1st latent component of first 5 channels of 7_2
    for c in range(num_channels_per_layer):
        sx = c
        stitched_image[sx*nx:(sx+1)*nx, sy*ny:(sy+1)*ny] = save_tmp_pdf_and_load(pdfs_SD_pca_mu,
                                                                                 pdfs_SD_pca_std,
                                                                                 pdfs_TD_pca,
                                                                                 z,
                                                                                 10*c,
                                                                                 savedir) # num_pca_latents = 10
    sy = sy+1

    return stitched_image

# ================================================================
# ================================================================
def save_tmp_pdf_and_load(pdfs_SD_mu,
                          pdfs_SD_std,
                          pdfs_TD,
                          x,
                          c,
                          savedir):
    
    plt.figure(figsize=[1.5,1.5])
    plt.plot(x, pdfs_SD_mu[c,:], 'b')
    plt.fill_between(x, pdfs_SD_mu[c,:] - 1*pdfs_SD_std[c,:], pdfs_SD_mu[c,:] + 1*pdfs_SD_std[c,:], alpha = 0.5)
    plt.plot(x, pdfs_TD[c,:], 'c')
    plt.savefig(savedir + '/tmp.png')
    plt.close()

    return io.imread(savedir + '/tmp.png')[:,:,0]

# ================================================================
# ================================================================
def write_cfs(step,
              summary_writer,
              sess,
              cfs_abs_summary,
              cfs_angle_summary,
              display_abs_pl,
              display_angle_pl,
              sd_cfs_,
              td_cfs_,
              savedir):

    # write to tensorboard
    summary_writer.add_summary(sess.run(cfs_abs_summary,
                                        feed_dict = {display_abs_pl: prepare_for_tensorboard(stitch_cfs(sd_cfs_,
                                                                                                        td_cfs_,
                                                                                                        savedir,
                                                                                                       'abs'))}), step)

    summary_writer.add_summary(sess.run(cfs_angle_summary,
                                        feed_dict = {display_angle_pl: prepare_for_tensorboard(stitch_cfs(sd_cfs_,
                                                                                                          td_cfs_,
                                                                                                          savedir,
                                                                                                          'angle'))}), step)

# ================================================================
# function to stitch all images of a particular iteration together
# ================================================================
def stitch_cfs(sd_cfs_, td_cfs_, savedir, abs_or_angle):
        
    nx = 150
    ny = 150

    # show first 5 channels for all the 7 layers (c1_1, c2_1, c3_1, c4_1, c5_1, c6_1, c7_1)
    stitched_image = np.zeros((5*nx, 7*ny), dtype = np.float32)

    sy = 0
    for delta in [0, 32, 96, 224, 480, 608, 672]:                
        for c in range(5):
            sx = c
            stitched_image[sx*nx:(sx+1)*nx, sy*ny:(sy+1)*ny] = save_tmp_cf_and_load(sd_cfs_, td_cfs_, delta+c, savedir, abs_or_angle)
        sy = sy+1

    return stitched_image

# ================================================================
# ================================================================
def save_tmp_cf_and_load(sd_cfs_, td_cfs_, c, savedir, abs_or_angle):
    
    plt.figure(figsize=[1.5,1.5])
    if abs_or_angle == 'abs':
        plt.plot(np.arange(sd_cfs_.shape[1]), np.abs(sd_cfs_[c,:]), 'b')
        plt.plot(np.arange(td_cfs_.shape[1]), np.abs(td_cfs_[c,:]), 'c--')
    elif abs_or_angle == 'angle':
        plt.plot(np.arange(sd_cfs_.shape[1]), np.angle(sd_cfs_[c,:]), 'b')
        plt.plot(np.arange(td_cfs_.shape[1]), np.angle(td_cfs_[c,:]), 'c--')
    plt.savefig(savedir + '/tmp.png')
    plt.close()

    return io.imread(savedir + '/tmp.png')[:,:,0]

# ==========================================================
# ==========================================================       
def save_patches(patches,
                 savepath,
                 ids = [-100, -100],
                 nc = 5,
                 nr = 5,
                 psize = 128):
    
    if ids == [-100, -100]:
        ids = np.random.randint(0, patches.shape[0], nc * nr)

    plt.figure(figsize=[nc*3, nr*3])
    for c in range(nc):     
        for r in range(nr): 
            plt.subplot(nc, nr, nc*c+r+1)
            plt.imshow(np.reshape(patches[ids[nc*c+r],:], [psize,psize]), cmap='gray')
            # plt.clim([0,1.1])
            plt.colorbar()
            plt.title('Mean: ' + str(np.round(np.mean(patches[ids[nc*c+r],:]), 2)))
    
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()

# ==========================================================
# ==========================================================       
def save_features(features,
                  savepath):
    
    nc = 4
    nr = 4

    plt.figure(figsize=[nc*3, nr*3])
    for c in range(nc):     
        for r in range(nr): 
            if nc*c+r < features.shape[0]:
                plt.subplot(nc, nr, nc*c+r+1)
                plt.imshow(features[nc*c+r, :, :], cmap='gray')
                # plt.clim([0, 1.1])
                plt.colorbar()
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()

# ==========================================================
# ==========================================================
def plot_scatter_pca_coefs(z_sd,
                           z_td,
                           savepath,
                           nc = 5,
                           nr = 5):

    plt.figure(figsize=[nc*3, nr*3])

    for c in range(nc):     
        for r in range(nr): 
            plt.subplot(nc, nr, nc*c+r+1)
            plt.scatter(z_sd[:,nc*c+r], np.zeros_like(z_sd[:,nc*c+r]), color=['blue'])
            plt.scatter(z_td[:,nc*c+r], np.zeros_like(z_td[:,nc*c+r]) + 1, color=['red'])
            plt.title('component ' + str(nc*c+r+1))
    
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()

# ==========================================================
# ==========================================================
def plot_histograms_pca_coefs(kdes_this_subject,
                              z_vals,
                              savepath,
                              nc = 3,
                              nr = 3):

    plt.figure(figsize=[nc*3, nr*3])

    for c in range(nc):     
        for r in range(nr):
            kde_this_dim = kdes_this_subject[nc*c+r, :]
            plt.subplot(nc, nr, nc*c+r+1)
            plt.plot(z_vals, kde_this_dim)
            plt.title('component ' + str(nc*c+r+1))
    
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()

# ==========================================================
# ==========================================================
def plot_scatter_pca_coefs_pairwise(z,
                                    savepath,
                                    nc = 4,
                                    nr = 4,
                                    set_limits = False):

    plt.figure(figsize=[nc*3, nr*3])

    for c in range(nc):     
        for r in range(nr): 
            if r >= c:
                plt.subplot(nc, nr, nc*c+r+1)
                plt.scatter(z[:,c], z[:,r], color=['blue'], marker=',')
                if set_limits == True:
                    plt.xlim([-15,15])
                    plt.ylim([-15,15])
                plt.title('component ' + str(c) + 'vs' + str(r))
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()

# ==========================================================
# ==========================================================
def visualize_principal_components(pcs,
                                   savepath,
                                   psize = 128,
                                   nc = 5,
                                   nr = 5):

    plt.figure(figsize=[nc*3, nr*3])
    for c in range(nc):     
        for r in range(nr): 
            plt.subplot(nc, nr, nc*c+r+1)
            plt.imshow(np.reshape(pcs[nc*c+r, :], [psize, psize]), cmap='gray')
            plt.colorbar()
            plt.title('component: ' + str(nc*c+r+1))
    
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()

# ==========================================================
# ==========================================================
def plot_kdes_for_latents(kdes_sd_subjects_tr,
                          kdes_sd_subjects_tt,
                          kdes_sd_subjects_vl,
                          kdes_td_subjects_ts,
                          z_vals,
                          savepath,
                          nc = 3,
                          nr = 3):

    plt.figure(figsize=[nc*3, nr*3])
    for c in range(nc):     
        r=0
        plt.subplot(nc, nr, nc*c+r+1)
        for s in range(kdes_sd_subjects_tt.shape[0]):
            plt.plot(z_vals, kdes_sd_subjects_tt[s, c, :], 'olive', linewidth=0.5)  
        for s in range(kdes_sd_subjects_tr.shape[0]):
            plt.plot(z_vals, kdes_sd_subjects_tr[s, c, :], 'blue', linewidth=0.5)  
        plt.title('pc' + str(c+1) + ', tr_v_tt')

        r=1
        plt.subplot(nc, nr, nc*c+r+1)
        for s in range(kdes_sd_subjects_vl.shape[0]):
            plt.plot(z_vals, kdes_sd_subjects_vl[s, c, :], 'green', linewidth=0.5)  
        for s in range(kdes_sd_subjects_tr.shape[0]):
            plt.plot(z_vals, kdes_sd_subjects_tr[s, c, :], 'blue', linewidth=0.5)  
        plt.title('pc' + str(c+1) + ', tr_v_vl')

        r=2
        plt.subplot(nc, nr, nc*c+r+1)
        for s in range(kdes_td_subjects_ts.shape[0]):
            plt.plot(z_vals, kdes_td_subjects_ts[s, c, :], 'red', linewidth=0.5)  
        for s in range(kdes_sd_subjects_tr.shape[0]):
            plt.plot(z_vals, kdes_sd_subjects_tr[s, c, :], 'blue', linewidth=0.5)  
        plt.title('pc' + str(c+1) + ', tr_v_ts')

    plt.savefig(savepath, bbox_inches='tight')
    plt.close()

    plt.figure(figsize=[nc*3, nr*3])
    for c in range(nc):     
        r=0
        plt.subplot(nc, nr, nc*c+r+1)
        tmp_mu = np.mean(kdes_sd_subjects_tt[:, c, :], 0)
        tmp_std = np.std(kdes_sd_subjects_tt[:, c, :], 0)
        plt.fill_between(z_vals, tmp_mu - 1*tmp_std, tmp_mu + 1*tmp_std, alpha = 0.5, color = 'olive')
        tmp_mu = np.mean(kdes_sd_subjects_tr[:, c, :], 0)
        tmp_std = np.std(kdes_sd_subjects_tr[:, c, :], 0)
        plt.fill_between(z_vals, tmp_mu - 1*tmp_std, tmp_mu + 1*tmp_std, alpha = 0.5, color = 'blue')
        plt.title('pc' + str(c+1) + ', tr_v_tt')

        r=1
        plt.subplot(nc, nr, nc*c+r+1)
        tmp_mu = np.mean(kdes_sd_subjects_vl[:, c, :], 0)
        tmp_std = np.std(kdes_sd_subjects_vl[:, c, :], 0)
        plt.fill_between(z_vals, tmp_mu - 1*tmp_std, tmp_mu + 1*tmp_std, alpha = 0.5, color = 'green')
        tmp_mu = np.mean(kdes_sd_subjects_tr[:, c, :], 0)
        tmp_std = np.std(kdes_sd_subjects_tr[:, c, :], 0)
        plt.fill_between(z_vals, tmp_mu - 1*tmp_std, tmp_mu + 1*tmp_std, alpha = 0.5, color = 'blue')
        plt.title('pc' + str(c+1) + ', tr_v_vl')

        r=2
        plt.subplot(nc, nr, nc*c+r+1)
        tmp_mu = np.mean(kdes_td_subjects_ts[:, c, :], 0)
        tmp_std = np.std(kdes_td_subjects_ts[:, c, :], 0)
        plt.fill_between(z_vals, tmp_mu - 1*tmp_std, tmp_mu + 1*tmp_std, alpha = 0.5, color = 'red')
        tmp_mu = np.mean(kdes_sd_subjects_tr[:, c, :], 0)
        tmp_std = np.std(kdes_sd_subjects_tr[:, c, :], 0)
        plt.fill_between(z_vals, tmp_mu - 1*tmp_std, tmp_mu + 1*tmp_std, alpha = 0.5, color = 'blue')
        plt.title('pc' + str(c+1) + ', tr_v_ts')
            
    plt.savefig(savepath[:-4] + '_summary.png', bbox_inches='tight')
    plt.close()

# ==========================================================
# ==========================================================
def plot_kdes_for_sd_and_td(kdes_td_subject,
                            kdes_sd_subjects,
                            z_vals,
                            savepath,
                            nc = 3,
                            nr = 3):

    plt.figure(figsize=[nc*3, nr*3])

    for c in range(nc):     
        for r in range(nr):
            plt.subplot(nc, nr, nc*c+r+1)
            for s in range(kdes_sd_subjects.shape[0]):
                plt.plot(z_vals, kdes_sd_subjects[s, nc*c+r, :], 'blue', linewidth=0.5)            
            plt.plot(z_vals, kdes_td_subject[nc*c+r, :], 'red', linewidth=0.5)
            plt.title('component ' + str(nc*c+r+1))
    
    plt.savefig(savepath, bbox_inches='tight')
    plt.close()