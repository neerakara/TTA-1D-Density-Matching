import model_zoo
import tensorflow as tf

# ====================================================
# normalizer architecture
# ====================================================
model_handle_normalizer = model_zoo.net2D_i2i
norm_kernel_size = 3
norm_num_hidden_layers = 2
norm_num_filters_per_layer = 16
norm_activation = 'rbf'
norm_batch_norm = False

# ====================================================
# settings of the i2l mapper 
# ====================================================
model_handle_i2l = model_zoo.unet2D_i2l_new

# ====================================================
# self-supervised helper networks
# ====================================================
model_handle_self_sup_ae = model_zoo.self_sup_autoencoder_like_yufan # model_zoo.self_sup_autoencoder
model_handle_self_sup_vae = model_zoo.self_sup_variational_autoencoder
model_handle_self_sup_dae = model_zoo.self_sup_denoising_autoencoder_3D
model_handle_adaptorAx = model_zoo.adaptor_Ax
model_handle_i2l_with_adaptors = model_zoo.unet2D_i2l_with_adaptors_new
model_handle_l2l = model_zoo.self_sup_denoising_autoencoder_3D

# ======================================================================
# training settings
# ======================================================================
batch_size = 16
learning_rate_tr = 1e-3
learning_rate_tl = 1e-4
optimizer_handle = tf.train.AdamOptimizer
loss_type = 'dice'
loss_type_l2l = 'dice'
debug = False

# ======================================================================
# max steps and frequencies for base network trainings from scratch
# ======================================================================
max_steps_tr = 30001 # initial training on SD
max_steps_tl = 5001
max_steps_ae = 25001
max_steps_dae = 50001
train_eval_frequency = 1000
val_eval_frequency = 1000   
save_frequency = 1000
summary_writing_frequency = 100

# ======================================================================
# data aug settings
# ======================================================================
sigma = 20
alpha = 1000
trans_min = -10
trans_max = 10
rot_min = -10
rot_max = 10
scale_min = 0.9
scale_max = 1.1
gamma_min = 0.5
gamma_max = 2.0
brightness_min = 0.0
brightness_max = 0.1
noise_min = 0.0
noise_max = 0.1

# ======================================================================
# ======================================================================
def get_dataset_dependent_params(train_dataset,
                                 test_dataset = ''):
    
    if train_dataset in ['CALTECH', 'STANFORD', 'HCPT1', 'HCPT2', 'IXI']:

        # =================================
        # size, resolution, etc.
        # =================================
        image_size = (256, 256)
        nlabels = 15
        target_resolution = (0.7, 0.7)
        downsampling_factor_x = 4
        downsampling_factor_y = 1
        downsampling_factor_z = 1
        
        if train_dataset == 'STANFORD':
            image_depth_tr = 132
        else:
            image_depth_tr = 256   

        if test_dataset == 'STANFORD':
            image_depth_ts = 132
        else:
            image_depth_ts = 256            

        # =================================
        # Whether to evaluate binary dice or over multiple classes
        # =================================
        whole_gland_results = False

        # =================================
        # number of TTA iterations
        # =================================
        tta_max_steps = 201 # Each step is an 'epoch' with num_batches = image_depth / args.b_size
        tta_model_saving_freq = 50
        tta_vis_freq = 10

        # =================================
        # Batch sizes for SD Gaussian / KDE computation
        # =================================
        b_size_compute_sd_pdfs = 16
        b_size_compute_sd_gaussians = 16

    elif train_dataset in ['BMC', 'RUNMC', 'UCL', 'HK', 'BIDMC', 'USZ']:

        # =================================
        # size, resolution, etc.
        # =================================
        nlabels = 3
        image_depth_tr = 32
        image_depth_ts = 32
        image_size = (256, 256)
        target_resolution = (0.625, 0.625)

        image_size_3d = (32, 256, 256)
        target_resolution_3d = (2.5, 0.625, 0.625)
        downsampling_factor_x = 1
        downsampling_factor_y = 1
        downsampling_factor_z = 1

        # =================================
        # Whether to evaluate binary dice or over multiple classes
        # =================================
        whole_gland_results = True

        # =================================
        # number of TTA iterations
        # =================================
        tta_max_steps = 1001 # Each step is an 'epoch' with num_batches = image_depth / args.b_size
        tta_model_saving_freq = 250
        tta_vis_freq = 10

        # =================================
        # Batch sizes for SD Gaussian / KDE computation
        # =================================
        # b_size_compute_sd_pdfs is set to a low value,
        # because the last incomplete batch is ignored in the SD KDE computation.
        # So we would like to cover almost the whole image with 'full' batches.
        b_size_compute_sd_pdfs = 2 
        b_size_compute_sd_gaussians = 0 # Use full images

    elif train_dataset in ['CSF', 'UHE', 'HVHD']:

        # =================================
        # size, resolution, etc.
        # =================================
        image_size = (256, 256)
        nlabels = 4
        target_resolution = (1.33, 1.33)
        downsampling_factor_x = 1
        downsampling_factor_y = 1
        downsampling_factor_z = 1
        image_depth_tr = 16
        image_depth_ts = 16

        # =================================
        # Whether to evaluate binary dice or over multiple classes
        # =================================
        whole_gland_results = False

        # =================================
        # number of TTA iterations
        # =================================
        tta_max_steps = 1001 # Each step is an 'epoch' with num_batches = image_depth / args.b_size
        tta_model_saving_freq = 50
        tta_vis_freq = 10

        # =================================
        # Batch sizes for SD Gaussian / KDE computation
        # =================================
        # b_size_compute_sd_pdfs is set to a low value,
        # because the last incomplete batch is ignored in the SD KDE computation.
        # So we would like to cover almost the whole image with 'full' batches.
        b_size_compute_sd_pdfs = 2 
        b_size_compute_sd_gaussians = 0 # Use full images

    elif train_dataset in ['VU', 'UMC', 'NUHS']:

        # =================================
        # size, resolution, etc.
        # =================================
        image_size = (256, 256)
        nlabels = 2
        target_resolution = (1.0, 1.0)
        downsampling_factor_x = 1
        downsampling_factor_y = 1
        downsampling_factor_z = 1
        image_depth_tr = 48
        image_depth_ts = 48

        # =================================
        # Whether to evaluate binary dice or over multiple classes
        # =================================
        whole_gland_results = False

        # =================================
        # number of TTA iterations
        # =================================
        tta_max_steps = 1001 # Each step is an 'epoch' with num_batches = image_depth / args.b_size
        tta_model_saving_freq = 50
        tta_vis_freq = 10

        # =================================
        # Batch sizes for SD Gaussian / KDE computation
        # =================================
        # b_size_compute_sd_pdfs is set to a low value,
        # because the last incomplete batch is ignored in the SD KDE computation.
        # So we would like to cover almost the whole image with 'full' batches.
        b_size_compute_sd_pdfs = 2 
        b_size_compute_sd_gaussians = 0 # Use full images

    elif train_dataset in ['site1', 'site2', 'site3', 'site4']:

        # =================================
        # size, resolution, etc.
        # =================================
        image_size = (200, 200)
        nlabels = 3
        target_resolution = (0.25, 0.25)
        downsampling_factor_x = 1
        downsampling_factor_y = 1
        downsampling_factor_z = 1
        image_depth_tr = 16
        image_depth_ts = 16

        # =================================
        # Whether to evaluate binary dice or over multiple classes
        # =================================
        whole_gland_results = False

        # =================================
        # number of TTA iterations
        # =================================
        tta_max_steps = 1001 # Each step is an 'epoch' with num_batches = image_depth / args.b_size
        tta_model_saving_freq = 50
        tta_vis_freq = 10

        # =================================
        # Batch sizes for SD Gaussian / KDE computation
        # =================================
        # b_size_compute_sd_pdfs is set to a low value,
        # because the last incomplete batch is ignored in the SD KDE computation.
        # So we would like to cover almost the whole image with 'full' batches.
        b_size_compute_sd_pdfs = 2 
        b_size_compute_sd_gaussians = 0 # Use full images

    return (image_size, # 0
            nlabels, # 1
            target_resolution, # 2
            image_depth_tr, # 3
            image_depth_ts, # 4
            whole_gland_results, # 5
            tta_max_steps, # 6
            tta_model_saving_freq, # 7
            tta_vis_freq, # 8
            b_size_compute_sd_pdfs, # 9
            b_size_compute_sd_gaussians, # 10
            image_size_3d, # 11 
            target_resolution_3d, # 12
            downsampling_factor_x, # 13
            downsampling_factor_y, # 14
            downsampling_factor_z) # 15

# ================================================================
# Function to make the name for the experiment run according to TTA parameters
# ================================================================
def make_tta_exp_name(args, tta_method = 'FoE'):
    
    if tta_method == 'FoE':
        exp_str = args.tta_string + args.PDF_TYPE # Gaussian / KDE / KDE_PCA
        
        # loss function
        if args.PDF_TYPE == 'KDE':
            exp_str = exp_str + '/Alpha' + str(args.KDE_ALPHA) # If KDE is used, what's the smoothness parameter?
        
        exp_str = exp_str + '/' + args.LOSS_TYPE # KL / 

        if args.LOSS_TYPE == 'KL':
            if args.KL_ORDER == 'TD_vs_SD':
                exp_str = exp_str + '_' + args.KL_ORDER
            elif args.KL_ORDER == 'SD_vs_TD': # default
                exp_str = exp_str

        if args.PCA_LAMBDA > 0.0:
            exp_str = exp_str + '/' + make_pca_dir_name(args)[:-1] + '_lambda' + str(args.PCA_LAMBDA)

        # optimization details
        exp_str = exp_str + '/Vars' + args.TTA_VARS 
        exp_str = exp_str + '_BS' + str(args.b_size) # TTA batch size
        exp_str = exp_str + '_FS' + str(args.feature_subsampling_factor) # Feature sub_sampling
        exp_str = exp_str + '_rand' + str(args.features_randomized) # If FS > 1 (random or uniform)

        if args.TTA_VARS in ['AdaptAx', 'AdaptAxAf']:
            exp_str = exp_str + 'lambda_spectral_' + str(args.lambda_spectral)
            if args.instance_norm_in_Ax == 0:
                exp_str = exp_str + '_no_IN_in_Ax'
            if args.train_Ax_first == 0:
                exp_str = exp_str + '_random_init_Ax'
        
        # Matching with mean over SD subjects or taking expectation wrt SD subjects
        exp_str = exp_str + '/SD_MATCH' + str(args.match_with_sd) 
        
        # learning rate parameters
        exp_str = exp_str + '/LR' + str(args.tta_learning_rate) # TTA Learning Rate
        exp_str = exp_str + '_SCH' + str(args.tta_learning_sch) # TTA LR schedule
        exp_str = exp_str + '_run' + str(args.tta_runnum) # TTA run number
            
        exp_str = exp_str + '/'

    elif tta_method == 'entropy_min':
        exp_str = args.tta_string + 'EntropyMin/'
        exp_str = exp_str + '/Vars' + args.TTA_VARS 
        exp_str = exp_str + '_BS' + str(args.b_size) # TTA batch size
        exp_str = exp_str + '_LR' + str(args.tta_learning_rate) # TTA Learning Rate
        exp_str = exp_str + '_SCH' + str(args.tta_learning_sch) # TTA LR schedule
        exp_str = exp_str + '_run' + str(args.tta_runnum) # TTA run number
        exp_str = exp_str + '/'

    elif tta_method == 'AE':
        exp_str = args.tta_string + args.tta_method + '/r' + str(args.ae_runnum) + '/'
        exp_str = exp_str + 'subjectwise/AEs_' + str(args.whichAEs) + '/'
        exp_str = exp_str + 'lambda_spectral_' + str(args.lambda_spectral) + '_vars' + args.TTA_VARS 
        if args.TTA_VARS in ['AdaptAx', 'AdaptAxAf']:
            if args.instance_norm_in_Ax == 0:
                exp_str = exp_str + '_no_IN_in_Ax'
            if args.train_Ax_first == 0:
                exp_str = exp_str + '_random_init_Ax'
        exp_str = exp_str + '/BS' + str(args.b_size) # TTA batch size
        exp_str = exp_str + '_accumgrad' + str(args.accum_gradients) # TTA accumulate gradients or not
        exp_str = exp_str + '_LR' + str(args.tta_learning_rate) # TTA Learning Rate
        exp_str = exp_str + '_SCH' + str(args.tta_learning_sch) # TTA LR schedule
        exp_str = exp_str + '_run' + str(args.tta_runnum) # TTA run number
        exp_str = exp_str + '/'

    elif tta_method == 'DAE':
        exp_str = args.tta_string + args.tta_method + '/r' + str(args.dae_runnum) + '/'
        exp_str = exp_str + 'subjectwise/' + 'vars' + args.TTA_VARS 
        exp_str = exp_str + '/BS' + str(args.b_size) # TTA batch size
        exp_str = exp_str + '_LR' + str(args.tta_learning_rate) # TTA Learning Rate
        exp_str = exp_str + '_SCH' + str(args.tta_learning_sch) # TTA LR schedule
        exp_str = exp_str + '_run' + str(args.tta_runnum) # TTA run number
        exp_str = exp_str + '/'

    return exp_str

# ================================================================
# Function to make the name for the experiment run according to TL parameters
# ================================================================
def make_tl_exp_name(args):
    
    exp_str = args.TL_STRING + args.test_dataset + '_cv' + str(args.test_cv_fold_num) + '_vars' + args.TL_VARS + '_run' + str(args.tl_runnum) + '/'

    return exp_str

# ================================================================
# Function to make the name for the directory where the PCA KDEs will be stored
# ================================================================
def make_pca_dir_name(args):
    
    dirname = 'pca_p' + str(args.PCA_PSIZE) 
    dirname = dirname + 's' + str(args.PCA_STRIDE)
    dirname = dirname + '_dim' + str(args.PCA_LATENT_DIM)
    dirname = dirname + '_' + args.PCA_LAYER + '_act_th' + str(args.PCA_THRESHOLD)
    if args.PDF_TYPE == 'KDE':
        dirname = dirname + '_kde_alpha' + str(args.PCA_KDE_ALPHA) + '/'
    elif args.PDF_TYPE == 'GAUSSIAN':
        dirname = dirname + '_gaussians/'

    return dirname

# ================================================================
# Function to make the name for the file containing SD Gaussian parameters
# ================================================================
def make_sd_gaussian_names(basepath, args):

    sd_gaussians_filename = basepath + 'sd_gaussians_subjectwise'
    sd_gaussians_filename = sd_gaussians_filename + '_subsample' + str(args.feature_subsampling_factor)
    sd_gaussians_filename = sd_gaussians_filename + '_rand' + str(args.features_randomized)

    return sd_gaussians_filename + '.npy'

# ================================================================
# Function to make the name for the file containing SD Gaussian parameters
# ================================================================
def make_sd_RP_gaussian_names(path_to_model,
                              b_size,
                              args):

    fname = path_to_model + 'sd_gaussians_' + args.before_or_after_bn + '_BN_subjectwise'

    if b_size != 0:
        fname = fname + '_bsize' + str(b_size)

    fname = fname + '_RP_psize' + str(args.PATCH_SIZE) + '_numproj' + str(args.NUM_RANDOM_FEATURES) + '.npy' 

    return fname

# ================================================================
# Function to make the name for the file containing SD KDEs
# ================================================================
def make_sd_kde_name(b_size,
                     alpha,
                     group_kde_params):

    kde_str = 'sd_kdes_subjectwise_bsize' + str(b_size) + '_alpha' + str(alpha)
    kde_str = kde_str + '_' + str(group_kde_params[0])
    kde_str = kde_str + '_' + str(group_kde_params[1])
    kde_str = kde_str + '_' + str(group_kde_params[2])

    return kde_str
