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
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5

# ===============================================================
# ===============================================================
def test_train_val_split(patient_id, # 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
                         cv_fold_number):
    
    if cv_fold_number == 1:
        if patient_id in [10, 9, 8]: return 'test'
        elif patient_id in [7, 6]: return 'validation'
        else: return 'train'
        
    elif cv_fold_number == 2:
        if patient_id in [7, 6, 5]: return 'test'
        elif patient_id in [4, 3]: return 'validation'
        else: return 'train'

    # used to accumulate evals of cv1 and cv2 together
    elif cv_fold_number == 3:
        if patient_id in [4, 3, 2]: return 'test'
        elif patient_id in [1, 10]: return 'validation'
        else: return 'train'

    # used to accumulate evals of cv1, cv2 abd cv3 together
    elif cv_fold_number == 4:
        if patient_id in [10, 9, 8, 7, 6, 5, 4, 3, 2]: return 'test'
        elif patient_id in [1]: return 'validation'
        else: return 'train'

# ===============================================================
# ===============================================================
def count_subjects_and_patient_ids_list(input_folder,
                                        sub_dataset,
                                        cv_fold_number = 1):

    num_subjects = {'train': 0, 'test': 0, 'validation': 0}       
    patient_ids_list = {'train': [], 'test': [], 'validation': []}

    # get the list of files in the input folder
    for f in os.listdir(input_folder):
        if sub_dataset in f and 'image' in f:
            imagepath = input_folder + f
            labelpath = input_folder + f[:11] + 'mask-r1.nii.gz'
            patid = f[8:10]

            # read the image
            image = utils.load_nii(imagepath)[0]
            label = utils.load_nii(labelpath)[0]

            print('-------------------')
            print(patid)
            print(image.shape)

            # assign a test/train/val tag to this image
            tt = test_train_val_split(int(patid), cv_fold_number)

            # add pat id and number of z-slices of this image
            patient_ids_list[tt].append(patid)
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
                 cv_fold_number):

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
                                                                         cv_fold_number)
        
    # =======================
    # set the number of slices according to what has been found from the previous function
    # =======================
    nz, nx, ny = size
    n_test = num_subjects['test']
    n_train = num_subjects['train']
    n_val = num_subjects['validation']
    print(n_test)
    print(n_train)
    print(n_val)

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

        if train_test == 'test' and n_test == 0:
            continue
        elif train_test == 'train' and n_train == 0:
            continue
        elif train_test == 'validation' and n_val == 0:
            continue

        write_buffer = 0
        counter_from = 0
        patient_counter = 0

        for patient_num in range(len(patient_ids_list[train_test])):

            patient_id = patient_ids_list[train_test][patient_num]

            # ======================
            # label path for this subject
            # ======================
            orig_lbl_path = input_folder + sub_dataset + '-sc' + patient_id + '-mask-r1.nii.gz'

            # ======================
            # load label
            # ======================
            lbl, _, lbl_header = utils.load_nii(img_path = orig_lbl_path)

            # ======================
            # add patient to list
            # ======================
            patient_counter += 1
            pat_names_list[train_test].append(patient_id)
                        
            # ======================
            # save original dimensions
            # ======================
            nx_list[train_test].append(lbl.shape[0])
            ny_list[train_test].append(lbl.shape[1])
            nz_list[train_test].append(lbl.shape[2])

            # ======================
            # save original resolution
            # ======================
            px_list[train_test].append(float(lbl_header.get_zooms()[0]))
            py_list[train_test].append(float(lbl_header.get_zooms()[1]))
            pz_list[train_test].append(float(lbl_header.get_zooms()[2]))

            # ======================
            # rescale in 3d
            # ======================
            scale_vector = [lbl_header.get_zooms()[0] / target_resolution[1],
                            lbl_header.get_zooms()[1] / target_resolution[2],
                            lbl_header.get_zooms()[2] / target_resolution[0]]
            
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
                                sub_dataset = 'site1', # site1 / site2 / site3 / site4
                                cv_fold_number = 1): 

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    data_file_name = 'data_3d_size_%s_res_%s_%s_cv%d.hdf5' % (size_str, res_str, sub_dataset, cv_fold_number)

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
                     cv_fold_number)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')

# ===============================================================
# ===============================================================
if __name__ == '__main__':
    input_folder = sys_config.orig_data_root_scgm
    preprocessing_folder = sys_config.preproc_folder_scgm

    data_scgm = load_and_maybe_process_data(input_folder,
                                            preprocessing_folder,
                                            (16, 200, 200),
                                            (5.0, 0.25, 0.25),
                                            force_overwrite=False,
                                            sub_dataset = 'site2',
                                            cv_fold_number = 1)

    print(data_scgm['labels_train'].shape)
    print(data_scgm['labels_validation'].shape)
    print(data_scgm['labels_test'].shape)
    
    print(data_scgm['px_validation'][:]); print(data_scgm['py_validation'][:]); print(data_scgm['pz_validation'][:])
    print(data_scgm['nx_validation'][:]); print(data_scgm['ny_validation'][:]); print(data_scgm['nz_validation'][:])

    print(data_scgm['px_test'][:]); print(data_scgm['py_test'][:]); print(data_scgm['pz_test'][:])
    print(data_scgm['nx_test'][:]); print(data_scgm['ny_test'][:]); print(data_scgm['nz_test'][:])

    print(data_scgm['px_train'][:]); print(data_scgm['py_train'][:]); print(data_scgm['pz_train'][:])
    print(data_scgm['nx_train'][:]); print(data_scgm['ny_train'][:]); print(data_scgm['nz_train'][:])

    # site2: px, py 0.5 | pz 7.5 | nx ny 320 | nz 10-15
    # site1: px, py 0.5 | pz 5.0 | nx ny 100 | nz 3
    # site3: px, py 0.25 | pz 2.5 | nx 650 ny 750 | nz 26-28
    # site4: px, py 0.285 | pz 5.0 | nx ny ~500 | nz 12-14

    data_scgm.close()