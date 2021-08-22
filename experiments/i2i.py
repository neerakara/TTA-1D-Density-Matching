import model_zoo
import tensorflow as tf

# ======================================================================
# train settings
# ======================================================================
train_dataset = 'RUNMC' # CALTECH / HCPT2 / HCPT1 / BMC / RUNMC
run_number = 1
tr_str = 'tr' + train_dataset
run_str = '_r' + str(run_number) + '/'
expname_i2l = tr_str + run_str + 'i2i2l/'

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

# ======================================================================
# test settings
# ======================================================================
if train_dataset == 'HCPT1':
    whole_gland_results = False
elif train_dataset == 'BMC' or train_dataset == 'RUNMC':
    whole_gland_results = True
normalize = True

# ====================================================
# TTA experiment identifier
# ====================================================
tta_string = 'tta_pdf_matching/'
simul_string = 'simulated_domain_shifts/adaBN_v2/'

# ======================================================================
# data settings
# ======================================================================
data_mode = '2D'
image_size = (256, 256)
image_depth_hcp = 256
image_depth_caltech = 256
image_depth_ixi = 256
image_depth_stanford = 132
nlabels_brain = 15
nlabels_prostate = 3
loss_type = 'dice'

# ======================================================================
# training settings
# ======================================================================
batch_size = 16
learning_rate = 1e-3
optimizer_handle = tf.train.AdamOptimizer
continue_run = False
debug = False

# ================================================================
# set dataset dependent hyperparameters
# ================================================================
if train_dataset in ['CALTECH', 'STANFORD', 'HCPT1', 'HCPT2', 'IXI']:
    nlabels = nlabels_brain
    target_resolution = (0.7, 0.7)
    downsampling_factor_x = 4
    downsampling_factor_y = 1
    downsampling_factor_z = 1
    image_depth = 256

elif train_dataset in ['BMC', 'RUNMC', 'PIRAD_ERC', 'PROMISE']:
    nlabels = nlabels_prostate
    target_resolution = (0.625, 0.625)
    image_depth = 32
    downsampling_factor_x = 1
    downsampling_factor_y = 1
    downsampling_factor_z = 1

image_size_downsampled = (int(image_depth / downsampling_factor_x), int(256 / downsampling_factor_y), int(256 / downsampling_factor_z))
batch_size_downsampled = int(batch_size / downsampling_factor_x)   

# ======================================================================
# max steps and frequencies for base network trainings from scratch
# ======================================================================
max_steps = 50001
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