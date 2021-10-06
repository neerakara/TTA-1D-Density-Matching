# 2D dataset loaders
import data.data_hcp as data_hcp
import data.data_abide as data_abide
import data.data_nci as data_nci
import data.data_promise as data_promise
import data.data_pirad_erc as data_pirad_erc
import data.data_mnms as data_mnms
import data.data_wmh as data_wmh
import data.data_scgm as data_scgm

# 3D dataset loaders
import data.data_nci_3d as data_nci_3d
import data.data_mnms_3d as data_mnms_3d
import data.data_scgm_3d as data_scgm_3d

# other imports
import logging
import config.system_paths as sys_config
import numpy as np

# ==================================================================   
# TRAINING DATA LOADER
# ==================================================================   
def load_labels_3d(train_dataset,
                   image_size,
                   target_resolution,
                   cv_fold_num = 1):

    # ================================================================
    # NCI
    # ================================================================
    if train_dataset in ['RUNMC', 'BMC']:
    
        logging.info('Reading NCI - ' + train_dataset + ' images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_nci)

        data_pros = data_nci_3d.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_nci,
                                                            preprocessing_folder = sys_config.preproc_folder_nci,
                                                            size = image_size,
                                                            target_resolution = target_resolution,
                                                            force_overwrite = False,
                                                            sub_dataset = train_dataset,
                                                            cv_fold_num = cv_fold_num)
        
        
        gttr = data_pros['labels_train']
        gtvl = data_pros['labels_validation']
        
        orig_data_res_x = data_pros['px_train'][:]
        orig_data_res_y = data_pros['py_train'][:]
        orig_data_res_z = data_pros['pz_train'][:]
        orig_data_siz_x = data_pros['nx_train'][:]
        orig_data_siz_y = data_pros['ny_train'][:]
        orig_data_siz_z = data_pros['nz_train'][:]

        num_train_subjects = orig_data_siz_z.shape[0] 

    # ================================================================
    # Cardiac (MnMs)
    # ================================================================
    elif train_dataset in ['CSF', 'UHE', 'HVHD']:
    
        logging.info('Reading - ' + train_dataset + ' labels...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_mnms)

        data_mnms = data_mnms_3d.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_mnms,
                                                             preprocessing_folder = sys_config.preproc_folder_mnms,
                                                             size = image_size,
                                                             target_resolution = target_resolution,
                                                             force_overwrite = False,
                                                             sub_dataset = train_dataset)
        
        
        gttr = data_mnms['labels_train']
        gtvl = data_mnms['labels_validation']
        
        orig_data_res_x = data_mnms['px_train'][:]
        orig_data_res_y = data_mnms['py_train'][:]
        orig_data_res_z = data_mnms['pz_train'][:]
        orig_data_siz_x = data_mnms['nx_train'][:]
        orig_data_siz_y = data_mnms['ny_train'][:]
        orig_data_siz_z = data_mnms['nz_train'][:]

        num_train_subjects = orig_data_siz_z.shape[0] 

    # ================================================================
    # Spine (SCGM)
    # ================================================================
    elif train_dataset in ['site2', 'site1', 'site3', 'site4']:
    
        logging.info('Reading - ' + train_dataset + ' labels...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_scgm)

        data_scgm = data_scgm_3d.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_scgm,
                                                             preprocessing_folder = sys_config.preproc_folder_scgm,
                                                             size = image_size,
                                                             target_resolution = target_resolution,
                                                             force_overwrite = False,
                                                             sub_dataset = train_dataset,
                                                             cv_fold_number = cv_fold_num)
                
        gttr = data_scgm['labels_train']
        gtvl = data_scgm['labels_validation']
        
        orig_data_res_x = data_scgm['px_train'][:]
        orig_data_res_y = data_scgm['py_train'][:]
        orig_data_res_z = data_scgm['pz_train'][:]
        orig_data_siz_x = data_scgm['nx_train'][:]
        orig_data_siz_y = data_scgm['ny_train'][:]
        orig_data_siz_z = data_scgm['nz_train'][:]

        num_train_subjects = orig_data_siz_z.shape[0] 

    return (gttr, # 0
            gtvl, # 1
            orig_data_res_x, # 2
            orig_data_res_y, # 3
            orig_data_res_z, # 4
            orig_data_siz_x, # 5
            orig_data_siz_y, # 6 
            orig_data_siz_z, # 7
            num_train_subjects) # 8 # 12

# ==================================================================   
# TRAINING DATA LOADER
# ==================================================================   
def load_training_data(train_dataset,
                       image_size,
                       target_resolution,
                       cv_fold_num = 1):

    # ================================================================
    # NCI
    # ================================================================
    if train_dataset in ['RUNMC', 'BMC']:
    
        logging.info('Reading NCI - ' + train_dataset + ' images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_nci)

        data_pros = data_nci.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_nci,
                                                         preprocessing_folder = sys_config.preproc_folder_nci,
                                                         size = image_size,
                                                         target_resolution = target_resolution,
                                                         force_overwrite = False,
                                                         sub_dataset = train_dataset,
                                                         cv_fold_num = cv_fold_num)
        
        imtr = data_pros['images_train']
        gttr = data_pros['labels_train']
        
        orig_data_res_x = data_pros['px_train'][:]
        orig_data_res_y = data_pros['py_train'][:]
        orig_data_res_z = data_pros['pz_train'][:]
        orig_data_siz_x = data_pros['nx_train'][:]
        orig_data_siz_y = data_pros['ny_train'][:]
        orig_data_siz_z = data_pros['nz_train'][:]

        num_train_subjects = orig_data_siz_z.shape[0] 

        imvl = data_pros['images_validation']
        gtvl = data_pros['labels_validation']
        orig_data_siz_z_val = data_pros['nz_validation'][:]
        num_val_subjects = orig_data_siz_z_val.shape[0] 

    elif train_dataset in ['UCL', 'BIDMC', 'HK']:
        logging.info('Reading' + train_dataset + ' images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_promise)

        data_pros = data_promise.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_promise,
                                                            preprocessing_folder = sys_config.preproc_folder_promise,
                                                            size = image_size,
                                                            target_resolution = target_resolution,
                                                            force_overwrite = False,
                                                            sub_dataset = train_dataset,
                                                            cv_fold_num = cv_fold_num)
        
        imtr = data_pros['images_train']
        gttr = data_pros['labels_train']
        
        orig_data_res_x = data_pros['px_train'][:]
        orig_data_res_y = data_pros['py_train'][:]
        orig_data_res_z = data_pros['pz_train'][:]
        orig_data_siz_x = data_pros['nx_train'][:]
        orig_data_siz_y = data_pros['ny_train'][:]
        orig_data_siz_z = data_pros['nz_train'][:]

        num_train_subjects = orig_data_siz_z.shape[0] 

        imvl = data_pros['images_validation']
        gtvl = data_pros['labels_validation']
        orig_data_siz_z_val = data_pros['nz_validation'][:]
        num_val_subjects = orig_data_siz_z_val.shape[0] 
        
    elif train_dataset in ['USZ']:
        
        logging.info('Reading PIRAD_ERC images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_pirad_erc)
        
        data_pros_train = data_pirad_erc.load_data(input_folder = sys_config.orig_data_root_pirad_erc,
                                                   preproc_folder = sys_config.preproc_folder_pirad_erc,
                                                   idx_start = 40,
                                                   idx_end = 68,
                                                   size = image_size,
                                                   target_resolution = target_resolution,
                                                   labeller = 'ek',
                                                   force_overwrite = False) 
        
        imtr = data_pros_train['images']
        gttr = data_pros_train['labels']
        orig_data_res_x = data_pros_train['px'][:]
        orig_data_res_y = data_pros_train['py'][:]
        orig_data_res_z = data_pros_train['pz'][:]
        orig_data_siz_x = data_pros_train['nx'][:]
        orig_data_siz_y = data_pros_train['ny'][:]
        orig_data_siz_z = data_pros_train['nz'][:]
        num_train_subjects = orig_data_siz_z.shape[0] 
        
        data_pros_val = data_pirad_erc.load_data(input_folder = sys_config.orig_data_root_pirad_erc,
                                                 preproc_folder = sys_config.preproc_folder_pirad_erc,
                                                 idx_start = 20,
                                                 idx_end = 40,
                                                 size = image_size,
                                                 target_resolution = target_resolution,
                                                 labeller = 'ek',
                                                 force_overwrite = False)

        imvl = data_pros_val['images']
        gtvl = data_pros_val['labels']
        orig_data_siz_z_val = data_pros_val['nz'][:]
        num_val_subjects = orig_data_siz_z_val.shape[0] 

    # ================================================================
    # CARDIAC (MNMS)
    # ================================================================
    elif train_dataset in ['HVHD', 'CSF', 'UHE']:
    
        logging.info('Reading MNMS - ' + train_dataset + ' images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_mnms)

        data_cardiac = data_mnms.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_mnms,
                                                             preprocessing_folder = sys_config.preproc_folder_mnms,
                                                             size = image_size,
                                                             target_resolution = target_resolution,
                                                             force_overwrite = False,
                                                             sub_dataset = train_dataset)
        
        imtr = data_cardiac['images_train']
        gttr = data_cardiac['labels_train']
        
        orig_data_res_x = data_cardiac['px_train'][:]
        orig_data_res_y = data_cardiac['py_train'][:]
        orig_data_res_z = data_cardiac['pz_train'][:]
        orig_data_siz_x = data_cardiac['nx_train'][:]
        orig_data_siz_y = data_cardiac['ny_train'][:]
        orig_data_siz_z = data_cardiac['nz_train'][:]

        num_train_subjects = orig_data_siz_z.shape[0] 

        imvl = data_cardiac['images_validation']
        gtvl = data_cardiac['labels_validation']
        orig_data_siz_z_val = data_cardiac['nz_validation'][:]
        num_val_subjects = orig_data_siz_z_val.shape[0] 

    # ================================================================
    # Brain lesions (WMH)
    # ================================================================
    elif train_dataset in ['UMC', 'NUHS']:
        data_brain_lesions = data_wmh.load_and_maybe_process_data(sys_config.orig_data_root_wmh,
                                                                  sys_config.preproc_folder_wmh,
                                                                  image_size,
                                                                  target_resolution,
                                                                  force_overwrite=False,
                                                                  sub_dataset = train_dataset,
                                                                  cv_fold_number = cv_fold_num,
                                                                  protocol = 'FLAIR')

        imtr = data_brain_lesions['images_train']
        gttr = data_brain_lesions['labels_train']
        
        orig_data_res_x = data_brain_lesions['px_train'][:]
        orig_data_res_y = data_brain_lesions['py_train'][:]
        orig_data_res_z = data_brain_lesions['pz_train'][:]
        orig_data_siz_x = data_brain_lesions['nx_train'][:]
        orig_data_siz_y = data_brain_lesions['ny_train'][:]
        orig_data_siz_z = data_brain_lesions['nz_train'][:]

        num_train_subjects = orig_data_siz_z.shape[0] 

        imvl = data_brain_lesions['images_validation']
        gtvl = data_brain_lesions['labels_validation']
        orig_data_siz_z_val = data_brain_lesions['nz_validation'][:]
        num_val_subjects = orig_data_siz_z_val.shape[0] 

    elif train_dataset in ['site1', 'site2', 'site3', 'site4']:
        data_gm = data_scgm.load_and_maybe_process_data(sys_config.orig_data_root_scgm,
                                                        sys_config.preproc_folder_scgm,
                                                        image_size,
                                                        target_resolution,
                                                        force_overwrite=False,
                                                        sub_dataset = train_dataset,
                                                        cv_fold_number = cv_fold_num)

        imtr = data_gm['images_train']
        gttr = data_gm['labels_train']
        
        orig_data_res_x = data_gm['px_train'][:]
        orig_data_res_y = data_gm['py_train'][:]
        orig_data_res_z = data_gm['pz_train'][:]
        orig_data_siz_x = data_gm['nx_train'][:]
        orig_data_siz_y = data_gm['ny_train'][:]
        orig_data_siz_z = data_gm['nz_train'][:]

        num_train_subjects = orig_data_siz_z.shape[0] 

        imvl = data_gm['images_validation']
        gtvl = data_gm['labels_validation']
        orig_data_siz_z_val = data_gm['nz_validation'][:]
        num_val_subjects = orig_data_siz_z_val.shape[0] 

    # ================================================================
    # HCP T1 / T2
    # ================================================================
    elif train_dataset in ['HCPT1', 'HCPT2']:

        logging.info('Reading ' + str(train_dataset) +  ' images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_hcp)
        
        data_brain_train = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                                preprocessing_folder = sys_config.preproc_folder_hcp,
                                                                idx_start = 0,
                                                                idx_end = 20,             
                                                                protocol = train_dataset[-2:],
                                                                size = image_size,
                                                                depth = 256,
                                                                target_resolution = target_resolution)
        
        imtr = data_brain_train['images']
        gttr = data_brain_train['labels']

        orig_data_res_x = data_brain_train['px'][:]
        orig_data_res_y = data_brain_train['py'][:]
        orig_data_res_z = data_brain_train['pz'][:]
        orig_data_siz_x = data_brain_train['nx'][:]
        orig_data_siz_y = data_brain_train['ny'][:]
        orig_data_siz_z = data_brain_train['nz'][:]

        num_train_subjects = orig_data_siz_z.shape[0] 

        data_brain_val = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                                preprocessing_folder = sys_config.preproc_folder_hcp,
                                                                idx_start = 20,
                                                                idx_end = 25,             
                                                                protocol = train_dataset[-2:],
                                                                size = image_size,
                                                                depth = 256,
                                                                target_resolution = target_resolution)
        
        imvl = data_brain_val['images']
        gtvl = data_brain_val['labels']
        orig_data_siz_z_val = data_brain_val['nz'][:]
        num_val_subjects = orig_data_siz_z_val.shape[0]
                
    elif train_dataset in ['CALTECH']:
        logging.info('Reading CALTECH images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_abide + 'CALTECH/')      
        data_brain_train = data_abide.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_abide,
                                                                  preprocessing_folder = sys_config.preproc_folder_abide,
                                                                  site_name = 'CALTECH',
                                                                  idx_start = 0,
                                                                  idx_end = 10,             
                                                                  protocol = 'T1',
                                                                  size = image_size,
                                                                  depth = 256,
                                                                  target_resolution = target_resolution)

        imtr = data_brain_train['images']
        gttr = data_brain_train['labels']

        orig_data_res_x = data_brain_train['px'][:]
        orig_data_res_y = data_brain_train['py'][:]
        orig_data_res_z = data_brain_train['pz'][:]
        orig_data_siz_x = data_brain_train['nx'][:]
        orig_data_siz_y = data_brain_train['ny'][:]
        orig_data_siz_z = data_brain_train['nz'][:]

        num_train_subjects = orig_data_siz_z.shape[0] 
        
        data_brain_val = data_abide.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_abide,
                                                                preprocessing_folder = sys_config.preproc_folder_abide,
                                                                site_name = 'CALTECH',
                                                                idx_start = 10,
                                                                idx_end = 15,             
                                                                protocol = 'T1',
                                                                size = image_size,
                                                                depth = 256,
                                                                target_resolution = target_resolution)
        imvl = data_brain_val['images']
        gtvl = data_brain_val['labels']
        orig_data_siz_z_val = data_brain_val['nz'][:]
        num_val_subjects = orig_data_siz_z_val.shape[0]
                
    return (imtr, # 0
            gttr, # 1
            orig_data_res_x, # 2
            orig_data_res_y, # 3
            orig_data_res_z, # 4
            orig_data_siz_x, # 5
            orig_data_siz_y, # 6 
            orig_data_siz_z, # 7
            num_train_subjects, # 8
            imvl, # 9
            gtvl, # 10
            orig_data_siz_z_val, # 11
            num_val_subjects) # 12

# ==================================================================   
# TEST DATA LOADER
# ==================================================================   
def load_testing_data(test_dataset,
                      cv_fold_num,
                      image_size,
                      target_resolution,
                      image_depth):

    # ================================================================
    # PROMISE
    # ================================================================
    if test_dataset in ['UCL', 'BIDMC', 'HK']:
        data_pros = data_promise.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_promise,
                                                            preprocessing_folder = sys_config.preproc_folder_promise,
                                                            size = image_size,
                                                            target_resolution = target_resolution,
                                                            force_overwrite = False,
                                                            sub_dataset = test_dataset,
                                                            cv_fold_num = cv_fold_num)
        
        imts = data_pros['images_test']
        gtts = data_pros['labels_test']
        orig_data_res_x = data_pros['px_test'][:]
        orig_data_res_y = data_pros['py_test'][:]
        orig_data_res_z = data_pros['pz_test'][:]
        orig_data_siz_x = data_pros['nx_test'][:]
        orig_data_siz_y = data_pros['ny_test'][:]
        orig_data_siz_z = data_pros['nz_test'][:]
        name_test_subjects = data_pros['patnames_test']
        num_test_subjects = orig_data_siz_z.shape[0] 
        ids = np.arange(num_test_subjects)

    # ================================================================
    # USZ
    # ================================================================
    elif test_dataset == 'USZ':

        image_depth = 32
        z_resolution = 2.5
        idx_start = 0
        idx_end = 20

        data_pros = data_pirad_erc.load_data(input_folder = sys_config.orig_data_root_pirad_erc,
                                            preproc_folder = sys_config.preproc_folder_pirad_erc,
                                            idx_start = idx_start,
                                            idx_end = idx_end,
                                            size = image_size,
                                            target_resolution = target_resolution,
                                            labeller = 'ek')
        
        imts = data_pros['images']
        gtts = data_pros['labels']
        orig_data_res_x = data_pros['px'][:]
        orig_data_res_y = data_pros['py'][:]
        orig_data_res_z = data_pros['pz'][:]
        orig_data_siz_x = data_pros['nx'][:]
        orig_data_siz_y = data_pros['ny'][:]
        orig_data_siz_z = data_pros['nz'][:]
        name_test_subjects = data_pros['patnames']
        num_test_subjects = 10 # orig_data_siz_z.shape[0] 
        ids = np.arange(idx_start, idx_end)

    # ================================================================
    # NCI
    # ================================================================
    elif test_dataset in ['BMC', 'RUNMC']:
    
        logging.info('Reading ' + test_dataset + ' images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_nci)
    
        data_pros = data_nci.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_nci,
                                                         preprocessing_folder = sys_config.preproc_folder_nci,
                                                         size = image_size,
                                                         target_resolution = target_resolution,
                                                         force_overwrite = False,
                                                         sub_dataset = test_dataset,
                                                         cv_fold_num = cv_fold_num)
        
        imts = data_pros['images_test']
        gtts = data_pros['labels_test']
        orig_data_res_x = data_pros['px_test'][:]
        orig_data_res_y = data_pros['py_test'][:]
        orig_data_res_z = data_pros['pz_test'][:]
        orig_data_siz_x = data_pros['nx_test'][:]
        orig_data_siz_y = data_pros['ny_test'][:]
        orig_data_siz_z = data_pros['nz_test'][:]
        name_test_subjects = data_pros['patnames_test']
        num_test_subjects = orig_data_siz_z.shape[0] 
        ids = np.arange(num_test_subjects)

    # ================================================================
    # CARDIAC (MNMS)
    # ================================================================
    elif test_dataset == 'HVHD' or test_dataset == 'CSF' or test_dataset == 'UHE':
    
        logging.info('Reading MNMS - ' + test_dataset + ' images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_mnms)

        data_cardiac = data_mnms.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_mnms,
                                                             preprocessing_folder = sys_config.preproc_folder_mnms,
                                                             size = image_size,
                                                             target_resolution = target_resolution,
                                                             force_overwrite = False,
                                                             sub_dataset = test_dataset)
        
        imts = data_cardiac['images_test']
        gtts = data_cardiac['labels_test']
        orig_data_res_x = data_cardiac['px_test'][:]
        orig_data_res_y = data_cardiac['py_test'][:]
        orig_data_res_z = data_cardiac['pz_test'][:]
        orig_data_siz_x = data_cardiac['nx_test'][:]
        orig_data_siz_y = data_cardiac['ny_test'][:]
        orig_data_siz_z = data_cardiac['nz_test'][:]
        name_test_subjects = data_cardiac['patnames_test']
        num_test_subjects = orig_data_siz_z.shape[0] 
        ids = np.arange(num_test_subjects)

    # ================================================================
    # Brain lesions (WMH)
    # ================================================================
    elif test_dataset == 'UMC' or test_dataset == 'NUHS':
        data_brain_lesions = data_wmh.load_and_maybe_process_data(sys_config.orig_data_root_wmh,
                                                                  sys_config.preproc_folder_wmh,
                                                                  image_size,
                                                                  target_resolution,
                                                                  force_overwrite=False,
                                                                  sub_dataset = test_dataset,
                                                                  cv_fold_number = cv_fold_num,
                                                                  protocol = 'FLAIR')

        imts = data_brain_lesions['images_test']
        gtts = data_brain_lesions['labels_test']
        orig_data_res_x = data_brain_lesions['px_test'][:]
        orig_data_res_y = data_brain_lesions['py_test'][:]
        orig_data_res_z = data_brain_lesions['pz_test'][:]
        orig_data_siz_x = data_brain_lesions['nx_test'][:]
        orig_data_siz_y = data_brain_lesions['ny_test'][:]
        orig_data_siz_z = data_brain_lesions['nz_test'][:]
        name_test_subjects = data_brain_lesions['patnames_test']
        num_test_subjects = orig_data_siz_z.shape[0] 
        ids = np.arange(num_test_subjects)

    # ================================================================
    # SPINE
    # ================================================================
    elif test_dataset == 'site1' or test_dataset == 'site2' or test_dataset == 'site3' or test_dataset == 'site4':
        data_spine = data_scgm.load_and_maybe_process_data(sys_config.orig_data_root_scgm,
                                                           sys_config.preproc_folder_scgm,
                                                           image_size,
                                                           target_resolution,
                                                           force_overwrite=False,
                                                           sub_dataset = test_dataset,
                                                           cv_fold_number = cv_fold_num)

        imts = data_spine['images_test']
        gtts = data_spine['labels_test']
        orig_data_res_x = data_spine['px_test'][:]
        orig_data_res_y = data_spine['py_test'][:]
        orig_data_res_z = data_spine['pz_test'][:]
        orig_data_siz_x = data_spine['nx_test'][:]
        orig_data_siz_y = data_spine['ny_test'][:]
        orig_data_siz_z = data_spine['nz_test'][:]
        name_test_subjects = data_spine['patnames_test']
        num_test_subjects = orig_data_siz_z.shape[0] 
        ids = np.arange(num_test_subjects)

    # ================================================================
    # HCP T1
    # ================================================================
    elif test_dataset == 'HCPT1':

        logging.info('Reading HCPT1 images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_hcp)

        idx_start = 50
        idx_end = 70
        
        data_brain = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                        preprocessing_folder = sys_config.preproc_folder_hcp,
                                                        idx_start = idx_start,
                                                        idx_end = idx_end,           
                                                        protocol = 'T1',
                                                        size = image_size,
                                                        depth = image_depth,
                                                        target_resolution = target_resolution)

        imts = data_brain['images']
        gtts = data_brain['labels']
        orig_data_res_x = data_brain['px'][:]
        orig_data_res_y = data_brain['py'][:]
        orig_data_res_z = data_brain['pz'][:]
        orig_data_siz_x = data_brain['nx'][:]
        orig_data_siz_y = data_brain['ny'][:]
        orig_data_siz_z = data_brain['nz'][:]
        name_test_subjects = data_brain['patnames']
        num_test_subjects = 10 # imts.shape[0] // image_depth
        ids = np.arange(idx_start, idx_end)

    # ================================================================
    # HCP T2
    # ================================================================
    elif test_dataset == 'HCPT2':

        logging.info('Reading HCPT2 images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_hcp)

        idx_start = 50
        idx_end = 70
        
        data_brain = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                        preprocessing_folder = sys_config.preproc_folder_hcp,
                                                        idx_start = idx_start,
                                                        idx_end = idx_end,           
                                                        protocol = 'T2',
                                                        size = image_size,
                                                        depth = image_depth,
                                                        target_resolution = target_resolution)

        imts = data_brain['images']
        gtts = data_brain['labels']
        orig_data_res_x = data_brain['px'][:]
        orig_data_res_y = data_brain['py'][:]
        orig_data_res_z = data_brain['pz'][:]
        orig_data_siz_x = data_brain['nx'][:]
        orig_data_siz_y = data_brain['ny'][:]
        orig_data_siz_z = data_brain['nz'][:]
        name_test_subjects = data_brain['patnames']
        num_test_subjects = 10 # imts.shape[0] // image_depth
        ids = np.arange(idx_start, idx_end)
        
    # ================================================================
    # ABIDE CALTECH T1
    # ================================================================
    elif test_dataset == 'CALTECH':
        logging.info('Reading CALTECH images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_abide + 'CALTECH/')

        idx_start = 16
        idx_end = 36         
        
        data_brain = data_abide.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_abide,
                                                            preprocessing_folder = sys_config.preproc_folder_abide,
                                                            site_name = 'CALTECH',
                                                            idx_start = idx_start,
                                                            idx_end = idx_end,             
                                                            protocol = 'T1',
                                                            size = image_size,
                                                            depth = image_depth,
                                                            target_resolution = target_resolution)

        imts = data_brain['images']
        gtts = data_brain['labels']
        orig_data_res_x = data_brain['px'][:]
        orig_data_res_y = data_brain['py'][:]
        orig_data_res_z = data_brain['pz'][:]
        orig_data_siz_x = data_brain['nx'][:]
        orig_data_siz_y = data_brain['ny'][:]
        orig_data_siz_z = data_brain['nz'][:]
        name_test_subjects = data_brain['patnames']
        num_test_subjects = 10 # imts.shape[0] // image_depth
        ids = np.arange(idx_start, idx_end)

    # ================================================================
    # ABIDE STANFORD T1
    # ================================================================
    elif test_dataset == 'STANFORD':
        logging.info('Reading STANFORD images...')    
        logging.info('Data root directory: ' + sys_config.orig_data_root_abide + 'STANFORD/')

        idx_start = 16
        idx_end = 36         
        
        data_brain = data_abide.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_abide,
                                                            preprocessing_folder = sys_config.preproc_folder_abide,
                                                            site_name = 'STANFORD',
                                                            idx_start = idx_start,
                                                            idx_end = idx_end,             
                                                            protocol = 'T1',
                                                            size = image_size,
                                                            depth = image_depth,
                                                            target_resolution = target_resolution)

        imts = data_brain['images']
        gtts = data_brain['labels']
        orig_data_res_x = data_brain['px'][:]
        orig_data_res_y = data_brain['py'][:]
        orig_data_res_z = data_brain['pz'][:]
        orig_data_siz_x = data_brain['nx'][:]
        orig_data_siz_y = data_brain['ny'][:]
        orig_data_siz_z = data_brain['nz'][:]
        name_test_subjects = data_brain['patnames']
        num_test_subjects = 10 # imts.shape[0] // image_depth
        ids = np.arange(idx_start, idx_end)

    return (imts,  # 0
            gtts,  # 1
            orig_data_res_x, # 2
            orig_data_res_y, # 3
            orig_data_res_z, # 4
            orig_data_siz_x, # 5
            orig_data_siz_y, # 6 
            orig_data_siz_z, # 7
            name_test_subjects, # 8
            num_test_subjects, # 9
            ids) # 10

# ================================================================
# ================================================================
def load_testing_data_wo_preproc(test_dataset_name,
                                 ids,
                                 sub_num,
                                 subject_name,
                                 image_depth):

    if test_dataset_name == 'HCPT1':
        # image will be normalized to [0,1]
        image_orig, labels_orig = data_hcp.load_without_size_preprocessing(input_folder = sys_config.orig_data_root_hcp,
                                                                           idx = ids[sub_num],
                                                                           protocol = 'T1',
                                                                           preprocessing_folder = sys_config.preproc_folder_hcp,
                                                                           depth = image_depth)
        
    elif test_dataset_name == 'HCPT2':
        # image will be normalized to [0,1]
        image_orig, labels_orig = data_hcp.load_without_size_preprocessing(input_folder = sys_config.orig_data_root_hcp,
                                                                           idx = ids[sub_num],
                                                                           protocol = 'T2',
                                                                           preprocessing_folder = sys_config.preproc_folder_hcp,
                                                                           depth = image_depth)

    elif test_dataset_name == 'CALTECH':
        # image will be normalized to [0,1]
        image_orig, labels_orig = data_abide.load_without_size_preprocessing(input_folder = sys_config.orig_data_root_abide,
                                                                             site_name = 'CALTECH',
                                                                             idx = ids[sub_num],
                                                                             depth = image_depth)

    elif test_dataset_name == 'STANFORD':
        # image will be normalized to [0,1]
        image_orig, labels_orig = data_abide.load_without_size_preprocessing(input_folder = sys_config.orig_data_root_abide,
                                                                             site_name = 'STANFORD',
                                                                             idx = ids[sub_num],
                                                                             depth = image_depth)

    elif test_dataset_name in ['BMC', 'RUNMC']:
        # image will be normalized to [0,1]
        image_orig, labels_orig = data_nci.load_without_size_preprocessing(sys_config.orig_data_root_nci,
                                                                           sys_config.preproc_folder_nci,
                                                                           test_dataset_name,
                                                                           cv_fold_num=1,
                                                                           train_test='test',
                                                                           idx=ids[sub_num])

    elif test_dataset_name == 'USZ':
        # image will be normalized to [0,1]
        image_orig, labels_orig = data_pirad_erc.load_without_size_preprocessing(sys_config.orig_data_root_pirad_erc,
                                                                                 subject_name,
                                                                                 labeller='ek')

    elif test_dataset_name in ['UCL', 'BIDMC', 'HK']:
        # image will be normalized to [0,1]
        image_orig, labels_orig = data_promise.load_without_size_preprocessing(sys_config.preproc_folder_promise,
                                                                               subject_name[4:6])

    elif test_dataset_name in ['CSF', 'UHE', 'HVHD']:
        # image will be normalized to [0,1]
        image_orig, labels_orig = data_mnms.load_without_size_preprocessing(sys_config.preproc_folder_mnms,
                                                                            subject_name)

    elif test_dataset_name in ['VU', 'UMC', 'NUHS']:
        # image will be normalized to [0,1]
        image_orig, labels_orig = data_wmh.load_without_size_preprocessing(sys_config.orig_data_root_wmh,
                                                                           test_dataset_name,
                                                                           subject_name,
                                                                           'FLAIR')

    elif test_dataset_name in ['site1', 'site2', 'site3', 'site4']:
        # image will be normalized to [0,1]
        image_orig, labels_orig = data_scgm.load_without_size_preprocessing(sys_config.orig_data_root_scgm,
                                                                            sys_config.preproc_folder_scgm,
                                                                            test_dataset_name,
                                                                            subject_name)

    return image_orig, labels_orig
