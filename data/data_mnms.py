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
def count_slices_and_patient_ids_list(input_folder,
                                      preproc_folder,
                                      sub_dataset,
                                      cv_fold_number = 1):

    num_slices = {'train': 0, 'test': 0, 'validation': 0}       
    patient_ids_list = {'train': [], 'test': [], 'validation': []}
    ed_es_id_list = {'train': [], 'test': [], 'validation': []}

    sub_dataset_ids, ed_time_ids, es_time_ids, train_test_val = read_sub_dataset_details(preproc_folder, sub_dataset)

    #
    for dirName, subdirList, fileList in os.walk(input_folder):               
    
        if len(fileList) == 2:

            patient_id = fileList[0][:6]

            if patient_id in sub_dataset_ids:

                imagepath = dirName + '/' + patient_id + '_sa.nii.gz'
                labelpath = dirName + '/' + patient_id + '_sa_gt.nii.gz'
                img = utils.load_nii(imagepath)[0]

                ed_time = ed_time_ids[np.where(sub_dataset_ids==patient_id)]
                es_time = es_time_ids[np.where(sub_dataset_ids==patient_id)]
                tt = train_test_val[np.where(sub_dataset_ids==patient_id)][0]
                
                patient_ids_list[tt].append(patient_id + '_ED')
                num_slices[tt] += img[:, :, :, ed_time].shape[2]
                ed_es_id_list[tt].append(ed_time)

                patient_ids_list[tt].append(patient_id + '_ES')
                num_slices[tt] += img[:, :, :, es_time].shape[2]
                ed_es_id_list[tt].append(es_time)

    return num_slices, patient_ids_list, ed_es_id_list

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
    num_slices, patient_ids_list, ed_es_id_list = count_slices_and_patient_ids_list(input_folder,
                                                                                    preproc_folder,
                                                                                    sub_dataset)
        
    # =======================
    # set the number of slices according to what has been found from the previous function
    # =======================
    nx, ny = size
    n_test = num_slices['test']
    n_train = num_slices['train']
    n_val = num_slices['validation']
    print(n_test)
    print(n_train)
    print(n_val)

    # =======================
    # Create datasets for images and labels
    # =======================
    data = {}
    for tt, num_points in zip(['test', 'train', 'validation'], [n_test, n_train, n_val]):

        if num_points > 0:
            data['images_%s' % tt] = hdf5_file.create_dataset("images_%s" % tt, [num_points] + list(size), dtype=np.float32)
            data['labels_%s' % tt] = hdf5_file.create_dataset("labels_%s" % tt, [num_points] + list(size), dtype=np.uint8)

    lbl_list = {'test': [], 'train': [], 'validation': []}
    img_list = {'test': [], 'train': [], 'validation': []}
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
            
            # ================================
            # do bias field correction
            # ================================
            orig_img_path = input_folder + patient_id[:-3] + '/' + patient_id[:-3] + '_sa.nii.gz'
            orig_lbl_path = input_folder + patient_id[:-3] + '/' + patient_id[:-3] + '_sa_gt.nii.gz'
            ed_es_img_path = preprocessing_folder + 'IndividualNIFTI/' + patient_id + '.nii.gz'
            ed_es_lbl_path = preprocessing_folder + 'IndividualNIFTI/' + patient_id + '_gt.nii.gz'
            ed_es_n4_img_path = preprocessing_folder + 'IndividualNIFTI/' + patient_id + '_n4.nii.gz'

            # If bias corrected image exists, load 3D (bias corrected) image and label
            if os.path.isfile(ed_es_n4_img_path):
                img = utils.load_nii(img_path = ed_es_n4_img_path)[0]
                lbl = utils.load_nii(img_path = ed_es_lbl_path)[0]
            else: # do bias correction first
                # load the orig 4D image and label
                orig_img = utils.load_nii(img_path = orig_img_path)[0] 
                orig_lbl = utils.load_nii(img_path = orig_lbl_path)[0] 
                
                # save the ED/ES 3D image and label
                utils.save_nii(img_path = ed_es_img_path, data = orig_img[:, :, :, ed_es_id], affine = np.eye(4))
                utils.save_nii(img_path = ed_es_lbl_path, data = orig_lbl[:, :, :, ed_es_id], affine = np.eye(4))

                # do bias correction for the ED/ES 3D image
                subprocess.call(["/cluster/home/nkarani/softwares/N4/N4_th", ed_es_img_path, ed_es_n4_img_path])

                # load 3D (bias corrected) image and label
                img = utils.load_nii(img_path = ed_es_n4_img_path)[0]
                lbl = utils.load_nii(img_path = ed_es_lbl_path)[0]
        
            patient_counter += 1
            pat_names_list[train_test].append(patient_id)

            # ================================    
            # normalize the image
            # ================================    
            img = utils.normalise_image(img, norm_type='div_by_max')     
                        
            # ================================    
            # save original dimensions
            # ================================    
            nx_list[train_test].append(lbl.shape[0])
            ny_list[train_test].append(lbl.shape[1])
            nz_list[train_test].append(lbl.shape[2])

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
            
            ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################
            scale_vector = [pixel_size[0] / target_resolution[0],
                            pixel_size[1] / target_resolution[1]]

            for zz in range(img.shape[2]):

                slice_img = np.squeeze(img[:, :, zz])
                img_rescaled = transform.rescale(slice_img,
                                                 scale_vector,
                                                 order=1,
                                                 preserve_range=True,
                                                 multichannel=False,
                                                 mode = 'constant')

                slice_lbl = np.squeeze(lbl[:, :, zz])
                lbl_rescaled = transform.rescale(slice_lbl,
                                                 scale_vector,
                                                 order=0,
                                                 preserve_range=True,
                                                 multichannel=False,
                                                 mode='constant')

                img_cropped = utils.crop_or_pad_slice_to_size(img_rescaled, nx, ny)
                lbl_cropped = utils.crop_or_pad_slice_to_size(lbl_rescaled, nx, ny)

                img_list[train_test].append(img_cropped)
                lbl_list[train_test].append(lbl_cropped)

                write_buffer += 1

                # Writing needs to happen inside the loop over the slices
                if write_buffer >= MAX_WRITE_BUFFER:

                    counter_to = counter_from + write_buffer
                    _write_range_to_hdf5(data, train_test, img_list, lbl_list, counter_from, counter_to)
                    _release_tmp_memory(img_list, lbl_list, train_test)

                    # reset stuff for next iteration
                    counter_from = counter_to
                    write_buffer = 0


        logging.info('Writing remaining data')
        counter_to = counter_from + write_buffer

        _write_range_to_hdf5(data, train_test, img_list, lbl_list, counter_from, counter_to)
        _release_tmp_memory(img_list, lbl_list, train_test)

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
                         img_list,
                         lbl_list,
                         counter_from,
                         counter_to):

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))

    img_arr = np.asarray(img_list[train_test], dtype=np.float32)
    lbl_arr = np.asarray(lbl_list[train_test], dtype=np.uint8)

    hdf5_data['images_%s' % train_test][counter_from:counter_to, ...] = img_arr
    hdf5_data['labels_%s' % train_test][counter_from:counter_to, ...] = lbl_arr

# ===============================================================
# Helper function to reset the tmp lists and free the memory
# ===============================================================
def _release_tmp_memory(img_list, lbl_list, train_test):
    
    img_list[train_test].clear()
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

    data_file_name = 'data_2d_size_%s_res_%s_%s.hdf5' % (size_str, res_str, sub_dataset)

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
# function to read a single subjects image and labels without any pre-processing
# ===============================================================
def load_without_size_preprocessing(preproc_folder,
                                    patient_id):
                    
    img_path = preproc_folder + 'IndividualNIFTI/' + patient_id + '.nii.gz'
    lbl_path = preproc_folder + 'IndividualNIFTI/' + patient_id + '_gt.nii.gz'

    img = utils.load_nii(img_path = img_path)[0]
    lbl = utils.load_nii(img_path = lbl_path)[0]
    
    # normalize the image
    img = utils.normalise_image(img, norm_type='div_by_max')
    
    return img, lbl

# ===============================================================
# ===============================================================
if __name__ == '__main__':
    input_folder = sys_config.orig_data_root_mnms
    preprocessing_folder = sys_config.preproc_folder_mnms        

    data_mnms = load_and_maybe_process_data(input_folder,
                                            preprocessing_folder,
                                            (256, 256),
                                            (1.33, 1.33),
                                            force_overwrite=False,
                                            sub_dataset = 'HVHD')

    print(data_mnms['images_train'].shape)
    print(data_mnms['labels_train'].shape)
    print(data_mnms['images_test'].shape)
    print(data_mnms['labels_test'].shape)
    print(data_mnms['images_validation'].shape)
    print(data_mnms['labels_validation'].shape)