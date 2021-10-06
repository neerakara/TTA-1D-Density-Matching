import os
import numpy as np
import logging
import gc
import re
import skimage.io as io
import SimpleITK as sitk
import h5py
from skimage import transform
import utils
import config.system_paths as sys_config
import subprocess
import csv
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5

# IDs of the different sub-datasets within the WMH dataset
VU_IDS = [100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 112, 113, 114, 115, 116, 126, 132, 137, 144]
UMC_IDS = [0, 2, 4, 6, 8, 11, 17, 19, 21, 23, 25, 27, 29, 31, 33, 35, 37, 39, 41, 49]
NUHS_IDS = [50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69]

# ===============================================================
# ===============================================================
def test_train_val_split(patient_id,
                         sub_dataset_ids,
                         cv_fold_number):
    
    if cv_fold_number == 1:
        if patient_id in sub_dataset_ids[-5:]: return 'test'
        elif patient_id in sub_dataset_ids[-10:-5]: return 'validation'
        else: return 'train'
        
    elif cv_fold_number == 2:
        if patient_id in sub_dataset_ids[-10:-5]: return 'test'
        elif patient_id in sub_dataset_ids[-15:-10]: return 'validation'
        else: return 'train'

    # used to accumulate evals of cv1 and cv2 together
    elif cv_fold_number == 3:
        if patient_id in sub_dataset_ids[-10:]: return 'test'
        elif patient_id in sub_dataset_ids[-15:-10]: return 'validation'
        else: return 'train'

# ===============================================================
# ===============================================================
def count_subjects_and_patient_ids_list(input_folder,
                                        sub_dataset,
                                        protocol,
                                        cv_fold_number = 1):

    num_subjects = {'train': 0, 'test': 0, 'validation': 0}       
    patient_ids_list = {'train': [], 'test': [], 'validation': []}

    if sub_dataset == 'VU':       
        sub_dataset_ids = VU_IDS
    elif sub_dataset == 'UMC':
        sub_dataset_ids = UMC_IDS
    elif sub_dataset == 'NUHS':
        sub_dataset_ids = NUHS_IDS

    # get the list of all patient dirs for this subdataset
    sub_dataset_folder = input_folder + sub_dataset + '/'
    list_patdirs_this_subdataset = [f.path for f in os.scandir(sub_dataset_folder) if f.is_dir()]
    list_patids_this_subdataset = []
    for pat_path in list_patdirs_this_subdataset:
        list_patids_this_subdataset.append(int(pat_path[pat_path.rfind('/')+1:]))
    patids_this_subdataset = np.sort(np.array(list_patids_this_subdataset))

    # go through each patient and read the number of slices in this patient
    for patid in patids_this_subdataset:               
    
        # assign a test/train/val tag to this image
        tt = test_train_val_split(patid, sub_dataset_ids, cv_fold_number)
        # add pat id and number of z-slices of this image
        patient_ids_list[tt].append(str(patid))
        num_subjects[tt] += 1

    return num_subjects, patient_ids_list

# ===============================================================
# ===============================================================
def prepare_data(input_folder,
                 preproc_folder, 
                 output_file,
                 size,
                 target_resolution,
                 sub_dataset,
                 cv_fold_number,
                 protocol):

    # =======================
    # create the hdf5 file where everything will be written
    # =======================
    hdf5_file = h5py.File(output_file, "w")

    # =======================
    # read all the images and count the number of slices along the append axis (the one with the lowest resolution)
    # =======================
    logging.info('Counting files and parsing meta data...')    
    num_subjects, patient_ids_list = count_subjects_and_patient_ids_list(input_folder,
                                                                         sub_dataset,
                                                                         protocol,
                                                                         cv_fold_number)
        
    # =======================
    # set the number of slices according to what has been found from the previous function
    # =======================
    nz, nx, ny = size
    n_test = num_subjects['test']
    n_train = num_subjects['train']
    n_val = num_subjects['validation']
    logging.info('number of test slices: ' + str(n_test))
    logging.info('number of train slices: ' + str(n_train))
    logging.info('number of validation slices: ' + str(n_val))

    # =======================
    # Create datasets for images and labels
    # =======================
    data = {}
    for tt, num_points in zip(['test', 'train', 'validation'], [n_test, n_train, n_val]):

        if num_points > 0:
            data['labels_%s' % tt] = hdf5_file.create_dataset("labels_%s" % tt, [num_points] + list(size), dtype=np.uint8)

    lbl_list = {'test': [], 'train': [], 'validation': []}
    nx_list = {'test': [], 'train': [], 'validation': []}
    ny_list = {'test': [], 'train': [], 'validation': []}
    nz_list = {'test': [], 'train': [], 'validation': []}
    px_list = {'test': [], 'train': [], 'validation': []}
    py_list = {'test': [], 'train': [], 'validation': []}
    pz_list = {'test': [], 'train': [], 'validation': []}
    pat_names_list = {'test': [], 'train': [], 'validation': []}              
                
    # =======================
    # read data of each subject, preprocess it and write to the hdf5 file
    # =======================
    logging.info('Parsing image files')
    for train_test in ['test', 'train', 'validation']:

        write_buffer = 0
        counter_from = 0
        patient_counter = 0

        for patient_num in range(len(patient_ids_list[train_test])):

            patient_id = patient_ids_list[train_test][patient_num]
            
            # ================================
            # read label
            # ================================
            patdir = input_folder + sub_dataset + '/' + patient_id + '/'
            labelpath = patdir + 'wmh.nii.gz'
            lbl = utils.load_nii(labelpath)[0]
        
            # ================================
            # save patient ID
            # ================================
            patient_counter += 1
            pat_names_list[train_test].append(patient_id)

            # ================================    
            # Other anomalies than WMH have been marked as '2'. Merging this with the background.
            # ================================    
            lbl[lbl!=1.0] = 0.0
                        
            # ================================    
            # save original dimensions
            # ================================    
            nx_list[train_test].append(lbl.shape[0])
            ny_list[train_test].append(lbl.shape[1])
            nz_list[train_test].append(lbl.shape[2])

            # ================================    
            # save original resolution
            # https://ieeexplore.ieee.org/document/8669968/media#media
            # ================================    
            if sub_dataset == 'VU': # 3D FLAIR sequence
                pixel_size = [1.21, 1.21, 1.30]
            elif sub_dataset == 'UMC': # 2D FLAIR sequence
                pixel_size = [0.96, 0.95, 3.00] 
            elif sub_dataset == 'NUHS': # 2D FLAIR sequence
                pixel_size = [1.00, 1.00, 3.00]
            px_list[train_test].append(float(pixel_size[0]))
            py_list[train_test].append(float(pixel_size[1]))
            pz_list[train_test].append(float(pixel_size[2]))
            
            # ======================
            # rescale in 3d
            # ======================
            scale_vector = [pixel_size[0] / target_resolution[1],
                            pixel_size[1] / target_resolution[2],
                            pixel_size[2] / target_resolution[0]]
            
            lbl_rescaled = transform.rescale(lbl,
                                             scale_vector,
                                             order=0,
                                             preserve_range=True,
                                             multichannel=False,
                                             mode='constant')

            print('rescaled 3D lbl shape: ' + str(lbl_rescaled.shape))
            
            # ======================
            # go through each z slice, crop or pad to a constant size and then append the resized 
            # ======================
            lbl_rescaled_cropped_xy = []
            for zz in range(lbl_rescaled.shape[2]):
                lbl_rescaled_cropped_xy.append(utils.crop_or_pad_slice_to_size(lbl_rescaled[:,:,zz], nx, ny))
            lbl_rescaled_cropped_xy = np.array(lbl_rescaled_cropped_xy)
            print('lbl_rescaled_cropped_xy.shape: ' + str(lbl_rescaled_cropped_xy.shape))
                
            # ======================
            # now, the z slices have been moved to the axis 0 position - as was happening in the 2d processing case
            # let's make the size along this axis the same for all subjects
            # ======================
            lbl_rescaled_cropped = utils.crop_or_pad_volume_to_size_along_x(lbl_rescaled_cropped_xy, nz)
            print('lbl_rescaled_cropped.shape: ' + str(lbl_rescaled_cropped.shape))
                
            lbl_list[train_test].append(lbl_rescaled_cropped)

            # ======================
            # write this subject's labels to HDF5 file
            # ======================
            _write_range_to_hdf5(data,
                                 train_test,
                                 lbl_list,
                                 patient_counter,
                                 patient_counter+1)
            
            _release_tmp_memory(lbl_list,
                                train_test)

    # Write the small datasets
    for tt in ['test', 'train', 'validation']:
        hdf5_file.create_dataset('nx_%s' % tt, data=np.asarray(nx_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('ny_%s' % tt, data=np.asarray(ny_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('nz_%s' % tt, data=np.asarray(nz_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('px_%s' % tt, data=np.asarray(px_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('py_%s' % tt, data=np.asarray(py_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('pz_%s' % tt, data=np.asarray(pz_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('patnames_%s' % tt, data=np.asarray(pat_names_list[tt], dtype="S10"))
    
    # After test train loop:
    hdf5_file.close()

# ===============================================================
# Helper function to write a range of data to the hdf5 datasets
# ===============================================================
def _write_range_to_hdf5(hdf5_data,
                         train_test,
                         lbl_list,
                         counter_from,
                         counter_to):

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))
    lbl_arr = np.asarray(lbl_list[train_test], dtype=np.uint8)
    hdf5_data['labels_%s' % train_test][counter_from:counter_to, ...] = lbl_arr

# ===============================================================
# Helper function to reset the tmp lists and free the memory
# ===============================================================
def _release_tmp_memory(lbl_list, train_test):
    
    lbl_list[train_test].clear()
    gc.collect()

# ===============================================================
# ===============================================================
def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                size,
                                target_resolution,
                                force_overwrite=False,
                                sub_dataset = 'UMC', # VU / UMC / NUHS
                                cv_fold_number = 1, # 1 / 2
                                protocol = 'FLAIR'):

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    data_file_name = 'data_3d_size_%s_res_%s_%s_%s_cv%d.hdf5' % (size_str, res_str, protocol, sub_dataset, cv_fold_number)

    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder,
                     preprocessing_folder,
                     data_file_path,
                     size,
                     target_resolution,
                     sub_dataset,
                     cv_fold_number,
                     protocol)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')

# ===============================================================
# ===============================================================
if __name__ == '__main__':
    
    input_folder = sys_config.orig_data_root_wmh
    preprocessing_folder = sys_config.preproc_folder_wmh
    protocol = 'FLAIR'
    sub_dataset = 'UMC'

    data_wmh = load_and_maybe_process_data(input_folder,
                                           preprocessing_folder,
                                           (48, 256, 256),
                                           (3.0, 1.0, 1.0),
                                           force_overwrite=False,
                                           sub_dataset = sub_dataset, # VU / UMC / NUHS
                                           cv_fold_number = 1,
                                           protocol = protocol)

    print(data_wmh['labels_train'].shape)
    print(data_wmh['labels_test'].shape)
    print(data_wmh['labels_validation'].shape)

    print(data_wmh['px_train'][:])
    print(data_wmh['py_train'][:])
    print(data_wmh['pz_train'][:])

    print(data_wmh['nx_train'][:])
    print(data_wmh['ny_train'][:])
    print(data_wmh['nz_train'][:])

    # UMC: pz - 3, px py ~ 1.0, nz - 48, nx, ny 240
    # NUHS: pz - 3, px py ~ 1.0, nz - 48, nx, ny 256, 232
    # VU: pz - 1.3, px py ~ 1.2, nz - 83, nx, ny 132, 256

    data_wmh.close()