# ===============================================================
# visualization functions
# ===============================================================
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
import numpy as np
import logging
from scipy.stats import norm
from skimage import color
from skimage import io

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
                      nlabels,
                      add_pixel_each_label=True,
                      cmap='tab20',
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
    
def plot_graph(a, b, save_path):
    plt.figure()
    plt.plot(a, b)
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
                          y_gt):
    
    # stitch images
    stitched_image = stitch_images(x, xn, y, y_gt)
    
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
def stitch_images(x, xn, y, y_gt):
        
    nx, ny = x.shape[1:]
    stitched_image = np.zeros((4*nx, 5*ny), dtype = np.float32)
    vis_ids = np.linspace(0, x.shape[0], 9, dtype=np.uint8)[2:7]

    for i in range(5):
        sx = 0; sy = i; stitched_image[sx*nx:(sx+1)*nx, sy*ny:(sy+1)*ny] = x[vis_ids[i], :, :]
        sx = 1; sy = i; stitched_image[sx*nx:(sx+1)*nx, sy*ny:(sy+1)*ny] = xn[vis_ids[i], :, :]
        sx = 2; sy = i; stitched_image[sx*nx:(sx+1)*nx, sy*ny:(sy+1)*ny] = y[vis_ids[i], :, :]
        sx = 3; sy = i; stitched_image[sx*nx:(sx+1)*nx, sy*ny:(sy+1)*ny] = y_gt[vis_ids[i], :, :]
    
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
                    logits_present,
                    nlabels,
                    deltas = [0, 32, 96, 224, 480, 608, 672],
                    num_channels = 5):

    # stitch images
    stitched_image = stitch_gaussians(sd_mu,
                                      sd_var,
                                      td_mu,
                                      td_var,
                                      savedir,
                                      logits_present,
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
                     logits_present,
                     nlabels,
                     deltas,
                     num_channels):
        
    nx = 150
    ny = 150

    if logits_present == 0:
        # show first 5 channels for all the 7 layers (c1_1, c2_1, c3_1, c4_1, c5_1, c6_1, c7_1)
        stitched_image = np.zeros((num_channels*nx, len(deltas)*ny), dtype = np.float32)
    else:
        # show logit distributions as well
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

    if logits_present == 1:
        delta = 704
        for c in range(nlabels):
            if c < 5:
                sx = c
                stitched_image[sx*nx:(sx+1)*nx, sy*ny:(sy+1)*ny] = save_tmp_and_load(sd_means,
                                                                                     sd_variances,
                                                                                     td_means,
                                                                                     td_variances,
                                                                                     delta+c,
                                                                                     savedir)

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
def write_pdfs(step,
               summary_writer,
               sess,
               pdfs_summary,
               display_pl,
               pdfs_SD_mu,
               pdfs_SD_std,
               pdfs_TD,
               x,
               savedir,
               deltas = [0, 32, 96, 224, 480, 608, 672],
               num_channels = 5):

    # stitch images
    stitched_image = stitch_pdfs(pdfs_SD_mu,
                                 pdfs_SD_std,
                                 pdfs_TD,
                                 x,
                                 savedir,
                                 deltas,
                                 num_channels)
    
    # make shape and type like tensorboard wants
    final_image = prepare_for_tensorboard(stitched_image)
    
    # write to tensorboard
    summary_writer.add_summary(sess.run(pdfs_summary, feed_dict = {display_pl: final_image}), step)

# ================================================================
# function to stitch all images of a particular iteration together
# ================================================================
def stitch_pdfs(pdfs_SD_mu,
                pdfs_SD_std,
                pdfs_TD,
                x,
                savedir,
                deltas,
                num_channels):
        
    nx = 150
    ny = 150

    # show first 5 channels for all the 7 layers (c1_1, c2_1, c3_1, c4_1, c5_1, c6_1, c7_1)
    stitched_image = np.zeros((num_channels*nx, len(deltas)*ny), dtype = np.float32)

    sy = 0
    for delta in deltas:
        for c in range(num_channels):
            sx = c
            stitched_image[sx*nx:(sx+1)*nx, sy*ny:(sy+1)*ny] = save_tmp_pdf_and_load(pdfs_SD_mu, pdfs_SD_std, pdfs_TD, x, delta+c, savedir)
        sy = sy+1

    return stitched_image

# ================================================================
# ================================================================
def save_tmp_pdf_and_load(pdfs_SD_mu, pdfs_SD_std, pdfs_TD, x, c, savedir):
    
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