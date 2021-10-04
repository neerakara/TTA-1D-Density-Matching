import os
import numpy as np
import logging
import gc
import h5py
from skimage import transform
import pydicom as dicom
import nrrd
import utils
import config.system_paths as sys_config
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5

# ===============================================================
# ===============================================================
def test_train_val_split(patient_id,
                         sub_dataset,
                         cv_fold_number):
    
    if cv_fold_number == 1:

        if sub_dataset == 'RUNMC':
            if patient_id < 16: # 26
                return 'train'
            elif patient_id < 21: # 31
                return 'validation'
            else:
                return 'test'

        if sub_dataset == 'BMC':
            if patient_id < 45: # 74
                return 'train'
            elif patient_id < 59: # 83
                return 'validation'
            else:
                return 'test'

# ===============================================================
# ===============================================================
def get_patient_folders(image_folder,
                        folder_base,
                        sub_dataset,
                        cv_fold_number):

    folder_list = {'train': [], 'test': [], 'validation': []}

    for folder in os.listdir(image_folder):
    
        if folder.startswith(folder_base + '-01'):
            patient_id = int(folder.split('-')[-1])            
            train_test = test_train_val_split(patient_id, sub_dataset, cv_fold_number)
            folder_list[train_test].append(os.path.join(image_folder, folder))

        # IGNORE -02 and -03 for now
        # elif folder.startswith(folder_base + '-02') or folder.startswith(folder_base + '-03'):
        #     folder_list['test'].append(os.path.join(image_folder, folder))

    return folder_list

# ===============================================================
# ===============================================================
def prepare_data(input_folder,
                 preprocessing_folder,
                 output_file,
                 size,
                 target_resolution,
                 sub_dataset,
                 cv_fold_num):

    # =======================
    # sub-dataset specific params
    # =======================
    if sub_dataset == 'RUNMC':
        image_folder = input_folder + 'Images/Prostate-3T/'
        label_folder = input_folder + 'Labels/Prostate-3T/'
        folder_base = 'Prostate3T'
    elif sub_dataset == 'BMC':
        image_folder = input_folder + 'Images/PROSTATE-DIAGNOSIS/'
        label_folder = input_folder + 'Labels/PROSTATE-DIAGNOSIS/'
        folder_base = 'ProstateDx'

    # =======================
    # open the file to be written into
    # =======================
    hdf5_file = h5py.File(output_file, "w")

    # =======================
    # count number of subjects and their paths
    # =======================
    logging.info('Counting files and parsing meta data...')
    folder_list = get_patient_folders(image_folder,
                                      folder_base,
                                      sub_dataset,
                                      cv_fold_num)
    
    nz, nx, ny = size
    n_test = len(folder_list['test'])
    n_train = len(folder_list['train'])
    n_val = len(folder_list['validation'])

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
    # loop through train-test-val split
    # =======================
    logging.info('Parsing image files')
    for train_test in ['test', 'train', 'validation']:

        patient_counter = 0

        # =======================
        # loop through all subjects
        # =======================
        for folder in folder_list[train_test]:           

            logging.info('================================')
            logging.info('Doing: %s' % folder)
            patname = folder_base + '-' + str(folder.split('-')[-2]) + '-' + str(folder.split('-')[-1])
            pat_names_list[train_test].append(patname)

            # =======================
            # collect pixel size info for this subject
            # =======================
            lstFilesDCM = []  # create an empty list            
            for dirName, subdirList, fileList in os.walk(folder):
                # fileList.sort()
                for filename in fileList:           
                    if ".dcm" in filename.lower():  # check whether the file's DICOM
                        lstFilesDCM.append(os.path.join(dirName, filename))
            # Get ref file
            RefDs = dicom.read_file(lstFilesDCM[0])
            # Load dimensions based on the number of rows, columns, and slices (along the Z axis)
            ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(lstFilesDCM))
            # Load spacing values (in mm)
            pixel_size = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
            px_list[train_test].append(float(RefDs.PixelSpacing[0]))
            py_list[train_test].append(float(RefDs.PixelSpacing[1]))
            pz_list[train_test].append(float(RefDs.SliceThickness))
            print('PixelDims')
            print(ConstPixelDims)
            print('PixelSpacing')
            print(pixel_size)

            # =======================
            # read the labels
            # =======================
            lbl_path = os.path.join(label_folder, folder.split('/')[-1] + '.nrrd')
            lbl, options = nrrd.read(lbl_path)
            # fix swapped axes in the segmentation labels
            lbl = np.swapaxes(lbl, 0, 1)

            # =======================
            # save original shapes
            # =======================
            nx_list[train_test].append(lbl.shape[0])
            ny_list[train_test].append(lbl.shape[1])
            nz_list[train_test].append(lbl.shape[2])
            print('lbl.shape')
            print(lbl.shape)
            
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
            print('lbl_rescaled.shape')
            print(lbl_rescaled.shape)
            
            # ==================================
            # go through each z slice, crop or pad to a constant size and then append the resized 
            # ==================================
            lbl_rescaled_cropped_xy = []
            for zz in range(lbl_rescaled.shape[2]):
                lbl_rescaled_cropped_xy.append(utils.crop_or_pad_slice_to_size(lbl_rescaled[:,:,zz], nx, ny))
            lbl_rescaled_cropped_xy = np.array(lbl_rescaled_cropped_xy)
            print('lbl_rescaled_cropped_xy.shape')
            print(lbl_rescaled_cropped_xy.shape)
                
            # ==================================
            # now, the z slices have been moved to the axis 0 position - as was happening in the 2d processing case
            # let's make the size along this axis the same for all subjects
            # ==================================
            lbl_rescaled_cropped = utils.crop_or_pad_volume_to_size_along_x(lbl_rescaled_cropped_xy, nz)
            print('lbl_rescaled_cropped.shape')
            print(lbl_rescaled_cropped.shape)
                
            lbl_list[train_test].append(lbl_rescaled_cropped)

            # write to file
            _write_range_to_hdf5(data,
                                 train_test,
                                 lbl_list,
                                 patient_counter,
                                 patient_counter+1)
            
            _release_tmp_memory(lbl_list,
                                train_test)
            
            # update counter
            patient_counter += 1

    # Write the small datasets
    for tt in ['test', 'train', 'validation']:
        hdf5_file.create_dataset('nz_%s' % tt, data=np.asarray(nz_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('ny_%s' % tt, data=np.asarray(ny_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('nx_%s' % tt, data=np.asarray(nx_list[tt], dtype=np.uint16))
        hdf5_file.create_dataset('pz_%s' % tt, data=np.asarray(pz_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('py_%s' % tt, data=np.asarray(py_list[tt], dtype=np.float32))
        hdf5_file.create_dataset('px_%s' % tt, data=np.asarray(px_list[tt], dtype=np.float32))
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
                                sub_dataset = 'RUNMC',
                                cv_fold_num = 1):

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    data_file_name = 'data_3d_size_%s_res_%s_cv_fold_%d_%s.hdf5' % (size_str, res_str, cv_fold_num, sub_dataset)

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
                     cv_fold_num)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')

# ===============================================================
# ===============================================================
if __name__ == '__main__':
    input_folder = sys_config.orig_data_root_nci
    preprocessing_folder = sys_config.preproc_folder_nci

    data_nci = load_and_maybe_process_data(input_folder,
                                           preprocessing_folder,
                                           (32, 256, 256),
                                           (2.5, 0.625, 0.625),
                                           force_overwrite=False,
                                           sub_dataset = 'RUNMC',
                                           cv_fold_num = 1)

    print(data_nci['labels_train'].shape)
    print(data_nci['labels_validation'].shape)
    print(data_nci['labels_test'].shape)
    data_nci.close()                                           
