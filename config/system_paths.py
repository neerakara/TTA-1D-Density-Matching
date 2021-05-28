import os
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# ==================================================================
# SET THESE PATHS MANUALLY #########################################
# ==================================================================

# ==================================================================
# project dirs
# ==================================================================
tensorboard_root = '/cluster/work/cvl/nkarani/tb/projects/dg_seg/methods/tta_abn/v1/'
project_root = '/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/'
data_root = '/cluster/work/cvl/nkarani/data/preprocessed/segmentation/'

# ==================================================================
# dirs where the pre-processed data is stored
# ==================================================================
preproc_folder_hcp = os.path.join(data_root,'HCP/')
preproc_folder_abide = os.path.join(data_root,'ABIDE/')
preproc_folder_nci = os.path.join(data_root,'NCI/')
preproc_folder_pirad_erc = os.path.join(data_root,'USZ/')
preproc_folder_promise = os.path.join(data_root,'PROMISE/')
preproc_folder_acdc = os.path.join(data_root,'ACDC/')

# ==================================================================
# define dummy paths for the original data storage. directly using pre-processed data for now
# ==================================================================
orig_data_root_acdc = ''
orig_data_root_nci = '/cluster/work/cvl/shared/bmicdatasets/original/Challenge_Datasets/NCI_Prostate/'
orig_data_root_promise = '/cluster/work/cvl/shared/bmicdatasets/original/Challenge_Datasets/Prostate_PROMISE12/TrainingData/'
orig_data_root_pirad_erc = '/cluster/work/cvl/shared/bmicdatasets/original/USZ/Prostate/'
orig_data_root_abide = '/cluster/work/cvl/shared/bmicdatasets/original/ABIDE/'
orig_data_root_hcp = '/cluster/work/cvl/nkarani/data/preprocessed/segmentation/HCP/'
