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
model_handle_i2l = model_zoo.unet2D_i2l

# ====================================================
# TTA experiment identifier
# ====================================================
simul_string = 'simulated_domain_shifts/adaBN_v2/'

# ======================================================================
# training settings
# ======================================================================
batch_size = 16
learning_rate = 1e-3
optimizer_handle = tf.train.AdamOptimizer
loss_type = 'dice'
continue_run = False
debug = False

# ======================================================================
# max steps and frequencies for base network trainings from scratch
# ======================================================================
max_steps = 30001
train_eval_frequency = 1000
val_eval_frequency = 1000
save_frequency = 1000
summary_writing_frequency = 100

# ======================================================================
# data aug settings
# ======================================================================
da_ratio = 0.25
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
def get_dataset_dependent_params(train_dataset, test_dataset):
    
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
        tta_max_steps = 251 # Each step is an 'epoch' with num_batches = image_depth / args.b_size
        tta_model_saving_freq = 50
        tta_vis_freq = 25 

        # =================================
        # =================================
        b_size_compute_sd_pdfs = 16
        b_size_compute_sd_gaussians = 16

    elif train_dataset in ['NCI', 'PIRAD_ERC', 'PROMISE']:

        # =================================
        # size, resolution, etc.
        # =================================
        image_size = (256, 256)
        nlabels = 3
        target_resolution = (0.625, 0.625)
        downsampling_factor_x = 1
        downsampling_factor_y = 1
        downsampling_factor_z = 1
        image_depth_tr = 32
        image_depth_ts = 32

        # =================================
        # Whether to evaluate binary dice or over multiple classes
        # =================================
        whole_gland_results = True

        # =================================
        # number of TTA iterations
        # =================================
        tta_max_steps = 1001 # Each step is an 'epoch' with num_batches = image_depth / args.b_size
        tta_model_saving_freq = 250
        tta_vis_freq = 50

        # =================================
        # =================================
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
            b_size_compute_sd_gaussians) # 10

# ================================================================
# Function to make the name for the experiment run according to TTA parameters
# ================================================================
def make_tta_exp_name(args):
    exp_str = args.tta_string + 'KDE' + str(args.KDE) # Whether using KDE as an intermediate step or not 
    if args.KDE == 1:
        exp_str = exp_str + '/Alpha' + str(args.alpha) # If KDE is used, what's the smoothness parameter?
    elif args.KDE == 0:
        exp_str = exp_str + '/' + str(args.before_or_after_bn) + '_BN' # Gaussians computed before (using params stored in BN layers) or after BN
    exp_str = exp_str + '/' + args.match_moments # Gaussian_KL / Full_KL / Full_CF_L2
    exp_str = exp_str + '/Vars' + args.tta_vars 
    exp_str = exp_str + '_BS' + str(args.b_size) # TTA batch size
    exp_str = exp_str + '_FS' + str(args.feature_subsampling_factor) # Feature sub_sampling
    exp_str = exp_str + '_rand' + str(args.features_randomized) # If FS > 1 (random or uniform)
    exp_str = exp_str + '/SD_MATCH' + str(args.match_with_sd) # Matching with mean over SD subjects or taking expectation wrt SD subjects
    exp_str = exp_str + '/LR' + str(args.tta_learning_rate) # TTA Learning Rate
    exp_str = exp_str + '_SCH' + str(args.tta_learning_sch) # TTA LR schedule
    if args.tta_init_from_scratch == 1:
        exp_str = exp_str + '/Reinit_before_TTA'
    exp_str = exp_str + '/'

    return exp_str
