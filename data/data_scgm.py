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
def count_slices_and_patient_ids_list(input_folder,
                                      sub_dataset,
                                      cv_fold_number = 1):

    num_slices = {'train': 0, 'test': 0, 'validation': 0}       
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
            num_slices[tt] += image.shape[2]

    return num_slices, patient_ids_list

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
    num_slices, patient_ids_list = count_slices_and_patient_ids_list(input_folder,
                                                                     sub_dataset,
                                                                     cv_fold_number)
        
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

            # paths for this subject
            orig_img_path = input_folder + sub_dataset + '-sc' + patient_id + '-image.nii.gz'
            orig_lbl_path = input_folder + sub_dataset + '-sc' + patient_id + '-mask-r1.nii.gz'

            if sub_dataset == 'site1':
                # load image
                img, _, img_header = utils.load_nii(img_path = orig_img_path)
            else:
                n4_img_path = preproc_folder + 'IndividualNIFTI/' + sub_dataset + '_' + patient_id + '_n4.nii.gz'
                # If not done already, do bias correction first
                if not os.path.isfile(n4_img_path):
                    subprocess.call(["/cluster/home/nkarani/softwares/N4/N4_th", orig_img_path, n4_img_path])    
                # load image
                img, _, img_header = utils.load_nii(img_path = n4_img_path)

            # load label
            lbl = utils.load_nii(img_path = orig_lbl_path)[0]
            # add patient to list
            patient_counter += 1
            pat_names_list[train_test].append(patient_id)

            # normalize the image
            img = utils.normalise_image(img, norm_type='div_by_max')     
                        
            # save original dimensions
            nx_list[train_test].append(lbl.shape[0])
            ny_list[train_test].append(lbl.shape[1])
            nz_list[train_test].append(lbl.shape[2])

            # save original resolution
            px_list[train_test].append(float(img_header.get_zooms()[0]))
            py_list[train_test].append(float(img_header.get_zooms()[1]))
            pz_list[train_test].append(float(img_header.get_zooms()[2]))
            
            ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################
            scale_vector = [img_header.get_zooms()[0] / target_resolution[0],
                            img_header.get_zooms()[1] / target_resolution[1]]
            print(img_header.get_zooms())
            print(scale_vector)                            

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
                                sub_dataset = 'site1', # site1 / site2 / site3 / site4
                                cv_fold_number = 1): 

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    data_file_name = 'data_2d_size_%s_res_%s_%s_cv%d.hdf5' % (size_str, res_str, sub_dataset, cv_fold_number)

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
# function to read a single subjects image and labels without any pre-processing
# ===============================================================
def load_without_size_preprocessing(input_folder,
                                    preproc_folder,
                                    sub_dataset,
                                    patient_id):
                    
    if sub_dataset == 'site1':
        img_path = input_folder + sub_dataset + '-sc' + patient_id + '-image.nii.gz'
    else:
        img_path = preproc_folder + 'IndividualNIFTI/' + sub_dataset + '_' + patient_id + '_n4.nii.gz'
    lbl_path = input_folder + sub_dataset + '-sc' + patient_id + '-mask-r1.nii.gz'

    img = utils.load_nii(img_path = img_path)[0]
    lbl = utils.load_nii(img_path = lbl_path)[0]
    
    # normalize the image
    img = utils.normalise_image(img, norm_type='div_by_max')
    
    return img, lbl

# ===============================================================
# ===============================================================
if __name__ == '__main__':
    input_folder = sys_config.orig_data_root_scgm
    preprocessing_folder = sys_config.preproc_folder_scgm

    data_scgm = load_and_maybe_process_data(input_folder,
                                            preprocessing_folder,
                                            (100, 100),
                                            (0.5, 0.5),
                                            force_overwrite=False,
                                            sub_dataset = 'site4',
                                            cv_fold_number = 1)

    print(data_scgm['images_train'].shape)
    print(data_scgm['labels_train'].shape)
    print(data_scgm['images_test'].shape)
    print(data_scgm['labels_test'].shape)
    print(data_scgm['images_validation'].shape)
    print(data_scgm['labels_validation'].shape)