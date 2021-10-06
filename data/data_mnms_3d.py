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

# IDs of the different sub-datasets within the PROMISE12 dataset

# ===============================================================
# ===============================================================
def read_sub_dataset_details(preproc_folder,
                             sub_dataset):
    
    if sub_dataset == 'CSF':
        details_file = preproc_folder + 'mnms_csf.csv'
        train_id_max = 31
        val_id_max = 41
    elif sub_dataset == 'UHE':
        details_file = preproc_folder + 'mnms_uhe.csv'
        train_id_max = 11
        val_id_max = 16
    elif sub_dataset == 'HVHD':
        details_file = preproc_folder + 'mnms_hvhd.csv'
        train_id_max = 56
        val_id_max = 66

    pat_ids_this_subdataset = []
    ed_time_idx = []
    es_time_idx = []
    train_test_val = []

    with open(details_file, newline='') as csvfile:

        csvlines = csv.reader(csvfile, delimiter=' ', quotechar='|')

        rownum = 1
        for row in csvlines:
            pat_ids_this_subdataset.append(row[0][:6])
            row = row[0][7:]
            ed_time_idx.append(int(row[:row.find(',')]))
            es_time_idx.append(int(row[row.find(',')+1:]))

            if rownum < train_id_max:
                train_test_val.append('train')
            elif rownum < val_id_max:
                train_test_val.append('validation')
            else:
                train_test_val.append('test')
            rownum = rownum + 1
    
    pat_ids_this_subdataset = np.array(pat_ids_this_subdataset)
    ed_time_idx = np.array(ed_time_idx)
    es_time_idx = np.array(es_time_idx)
    train_test_val = np.array(train_test_val)

    return pat_ids_this_subdataset, ed_time_idx, es_time_idx, train_test_val

# ===============================================================
# ===============================================================
def count_subjects_and_patient_ids_list(input_folder,
                                        preproc_folder,
                                        sub_dataset,
                                        cv_fold_number = 1):

    num_subjects = {'train': 0, 'test': 0, 'validation': 0}       
    patient_ids_list = {'train': [], 'test': [], 'validation': []}
    ed_es_id_list = {'train': [], 'test': [], 'validation': []}

    sub_dataset_ids, ed_time_ids, es_time_ids, train_test_val = read_sub_dataset_details(preproc_folder, sub_dataset)

    #
    for dirName, subdirList, fileList in os.walk(input_folder):               
    
        if len(fileList) == 2:

            patient_id = fileList[0][:6]

            if patient_id in sub_dataset_ids:

                # imagepath = dirName + '/' + patient_id + '_sa.nii.gz'
                # labelpath = dirName + '/' + patient_id + '_sa_gt.nii.gz'
                # img = utils.load_nii(imagepath)[0]

                ed_time = ed_time_ids[np.where(sub_dataset_ids==patient_id)]
                es_time = es_time_ids[np.where(sub_dataset_ids==patient_id)]
                tt = train_test_val[np.where(sub_dataset_ids==patient_id)][0]
                
                patient_ids_list[tt].append(patient_id + '_ED')
                num_subjects[tt] += 1 # img[:, :, :, ed_time].shape[2]
                ed_es_id_list[tt].append(ed_time)

                patient_ids_list[tt].append(patient_id + '_ES')
                num_subjects[tt] += 1 # img[:, :, :, es_time].shape[2]
                ed_es_id_list[tt].append(es_time)

    return num_subjects, patient_ids_list, ed_es_id_list

# ===============================================================
# ===============================================================
def prepare_data(input_folder,
                 preproc_folder, 
                 output_file,
                 size,
                 target_resolution,
                 sub_dataset):

    # =======================
    # create the hdf5 file where everything will be written
    # =======================
    hdf5_file = h5py.File(output_file, "w")

    # =======================
    # read all the images and count the number of slices along the append axis (the one with the lowest resolution)
    # =======================
    logging.info('Counting files and parsing meta data...')    
    num_subjects, patient_ids_list, ed_es_id_list = count_subjects_and_patient_ids_list(input_folder,
                                                                                        preproc_folder,
                                                                                        sub_dataset)
        
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

        write_buffer = 0
        counter_from = 0
        patient_counter = 0

        for patient_num in range(len(patient_ids_list[train_test])):

            patient_id = patient_ids_list[train_test][patient_num]
            ed_es_id = ed_es_id_list[train_test][patient_num][0]

            # ======================
            # save patient ID
            # ======================
            pat_names_list[train_test].append(patient_id)
            
            # ================================
            # load the label file
            # ================================
            orig_lbl_path = input_folder + patient_id[:-3] + '/' + patient_id[:-3] + '_sa_gt.nii.gz'
            ed_es_lbl_path = preprocessing_folder + 'IndividualNIFTI/' + patient_id + '_gt.nii.gz'

            # If the 3D label image exists, load it
            if os.path.isfile(ed_es_lbl_path):
                lbl = utils.load_nii(img_path = ed_es_lbl_path)[0]
                
            else: # load the 4D label image, save the 3D volume at the required time index and load the 3D label
                # load the orig 4D image and label
                orig_lbl = utils.load_nii(img_path = orig_lbl_path)[0] 
                # save the ED/ES 3D image and label
                utils.save_nii(img_path = ed_es_lbl_path, data = orig_lbl[:, :, :, ed_es_id], affine = np.eye(4))
                # load 3D label
                lbl = utils.load_nii(img_path = ed_es_lbl_path)[0]
                                
            # ================================    
            # save original dimensions
            # ================================    
            nx_list[train_test].append(lbl.shape[0])
            ny_list[train_test].append(lbl.shape[1])
            nz_list[train_test].append(lbl.shape[2])

            print('orig 3D lbl shape: ' + str(lbl.shape))

            # ================================    
            # save original resolution
            # ================================    
            if sub_dataset == 'CSF':
                pixel_size = [1.20, 1.20, 9.90]
            elif sub_dataset == 'UHE':
                pixel_size = [1.45, 1.45, 9.90]
            elif sub_dataset == 'HVHD':
                pixel_size = [1.32, 1.32, 9.20]
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
            
            # ======================
            # update counter
            # ======================
            patient_counter += 1

    # ======================
    # Write the small datasets (original sizes, resolution and patient IDs)
    # ======================
    for tt in ['test', 'train', 'validation']:
        hdf5_file.create_dataset('nx_%s' % tt, data=np.asarray(nx_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('ny_%s' % tt, data=np.asarray(ny_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('nz_%s' % tt, data=np.asarray(nz_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('px_%s' % tt, data=np.asarray(px_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('py_%s' % tt, data=np.asarray(py_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('pz_%s' % tt, data=np.asarray(pz_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('patnames_%s' % tt, data=np.asarray(pat_names_list[tt], dtype="S10"))

    # ======================    
    # After test train loop:
    # ======================
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
                                sub_dataset = 'CSF'): # CSF / UHE / HVHD

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    data_file_name = 'data_3d_size_%s_res_%s_%s.hdf5' % (size_str, res_str, sub_dataset)

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
                     sub_dataset)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')

# ===============================================================
# ===============================================================
if __name__ == '__main__':
    input_folder = sys_config.orig_data_root_mnms
    preprocessing_folder = sys_config.preproc_folder_mnms        

    data_mnms = load_and_maybe_process_data(input_folder,
                                            preprocessing_folder,
                                            (16, 256, 256), # (16, 256, 256)
                                            (9.9, 1.33, 1.33), # (9.9, 1.33, 1.33)
                                            force_overwrite=False,
                                            sub_dataset = 'HVHD')

    print(data_mnms['labels_train'].shape)
    print(data_mnms['labels_validation'].shape)
    print(data_mnms['labels_test'].shape)    
    
    # nz (max): CSF ~ 12, HVHD ~ 16, UHE ~ 13
    # nz (median): CSF ~ 10, HVHD ~ 12, UHE ~ 11
    # pz (all): CSF ~ 9.9, HVHD ~ 9.2, UHE ~ 9.9 (is number ko ghatana -> matlab resolution badhana -> matlab upsample karna) (dikkat downsampling me aa sakti hai -> matlab is number ko badhane me aa sakti hai)
    # px (all): CSF ~ 1.2, HVHD ~ 1.32, UHE ~ 1.45

    data_mnms.close()