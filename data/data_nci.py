import os
import numpy as np
import logging
import gc
import h5py
from skimage import transform
import utils
import config.system_paths as sys_config
import pydicom as dicom
import nrrd
import subprocess
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
            if patient_id < 16:
                return 'train'
            elif patient_id < 21:
                return 'validation'
            else:
                return 'test'

        if sub_dataset == 'BMC':
            if patient_id < 45:
                return 'train'
            elif patient_id < 59:
                return 'validation'
            else:
                return 'test'

# ===============================================================
# ===============================================================
def count_slices(image_folder,
                 folder_base,
                 sub_dataset,
                 cv_fold_number):

    num_slices = {'train': 0,
                  'test': 0,
                  'validation': 0}

    for folder in os.listdir(image_folder):
        
        if folder.startswith(folder_base + '-01'):
            patient_id = int(folder.split('-')[-1])        
            for _, _, fileList in os.walk(os.path.join(image_folder, folder)):
                for filename in fileList:
                    if filename.lower().endswith('.dcm'):  # check whether the file's DICOM
                        train_test = test_train_val_split(patient_id,
                                                          sub_dataset,
                                                          cv_fold_number)
                        num_slices[train_test] += 1

        elif folder.startswith(folder_base + '-02') or folder.startswith(folder_base + '-03'):
            for _, _, fileList in os.walk(os.path.join(image_folder, folder)):
                for filename in fileList:
                    if filename.lower().endswith('.dcm'):  # check whether the file's DICOM
                        num_slices['test'] += 1

    return num_slices

# ===============================================================
# ===============================================================
def get_patient_folders(image_folder,
                        folder_base,
                        sub_dataset,
                        cv_fold_number):

    folder_list = {'train': [],
                   'test': [],
                   'validation': []}

    for folder in os.listdir(image_folder):
    
        if folder.startswith(folder_base + '-01'):
            patient_id = int(folder.split('-')[-1])            
            train_test = test_train_val_split(patient_id, sub_dataset, cv_fold_number)
            folder_list[train_test].append(os.path.join(image_folder, folder))

        elif folder.startswith(folder_base + '-02') or folder.startswith(folder_base + '-03'):
            folder_list['test'].append(os.path.join(image_folder, folder))

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
    # =======================
    hdf5_file = h5py.File(output_file, "w")

    # =======================
    # =======================
    logging.info('Counting files and parsing meta data...')
    folder_list = get_patient_folders(image_folder,
                                      folder_base,
                                      sub_dataset,
                                      cv_fold_num)
    
    num_slices = count_slices(image_folder,
                              folder_base,
                              sub_dataset,
                              cv_fold_num)
    
    nx, ny = size
    n_test = num_slices['test']
    n_train = num_slices['train']
    n_val = num_slices['validation']

    # =======================
    # Create datasets for images and masks
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
    # =======================
    logging.info('Parsing image files')
    for train_test in ['test', 'train', 'validation']:

        write_buffer = 0
        counter_from = 0

        patient_counter = 0

        for folder in folder_list[train_test]:

            patient_counter += 1

            logging.info('================================')
            logging.info('Doing: %s' % folder)
            patname = folder_base + '-' + str(folder.split('-')[-2]) + '-' + str(folder.split('-')[-1])
            pat_names_list[train_test].append(patname)

            # Make a list of all dicom files in this folder
            listFilesDCM = []  # create an empty list
            for dirName, subdirList, fileList in os.walk(folder):
                for filename in fileList:
                    if ".dcm" in filename.lower():  # check whether the file's DICOM
                        listFilesDCM.append(os.path.join(dirName, filename))

            # Get a reference dicom file and extract info such as number of rows, columns, and slices (along the Z axis)
            RefDs = dicom.read_file(listFilesDCM[0])
            ConstPixelDims = (int(RefDs.Rows), int(RefDs.Columns), len(listFilesDCM))
            pixel_size = (float(RefDs.PixelSpacing[0]), float(RefDs.PixelSpacing[1]), float(RefDs.SliceThickness))
            px_list[train_test].append(float(RefDs.PixelSpacing[0]))
            py_list[train_test].append(float(RefDs.PixelSpacing[1]))
            pz_list[train_test].append(float(RefDs.SliceThickness))

            print('PixelDims')
            print(ConstPixelDims)
            print('PixelSpacing')
            print(pixel_size)

            # The array is sized based on 'ConstPixelDims'
            img = np.zeros(ConstPixelDims, dtype=RefDs.pixel_array.dtype)

            # loop through all the DICOM files
            for filenameDCM in listFilesDCM:

                # read the file
                ds = dicom.read_file(filenameDCM)

                # ======
                # store the raw image data
                # img[:, :, listFilesDCM.index(filenameDCM)] = ds.pixel_array
                # index number field is not set correctly!
                # instead instance number is the slice number.
                # ======
                img[:, :, ds.InstanceNumber - 1] = ds.pixel_array
                
            # ================================
            # save as nifti, this sets the affine transformation as an identity matrix
            # ================================    
            nifti_img_path = preprocessing_folder + 'Individual_NIFTI/' + patname
            utils.save_nii(img_path = nifti_img_path + '_img.nii.gz', data = img, affine = np.eye(4))
    
            # ================================
            # do bias field correction
            # ================================
            input_img = nifti_img_path + '_img.nii.gz'
            output_img = nifti_img_path + '_img_n4.nii.gz'
            # If bias corrected image does not exist, do it now
            if os.path.isfile(output_img):
                img = utils.load_nii(img_path = output_img)[0]
            else:
                subprocess.call(["/cluster/home/nkarani/softwares/N4/N4_th", input_img, output_img])
                img = utils.load_nii(img_path = output_img)[0]

            # ================================    
            # normalize the image
            # ================================    
            img = utils.normalise_image(img, norm_type='div_by_max')

            # ================================    
            # read the labels
            # ================================    
            print(folder.split('/')[-1])
            lbl_path = os.path.join(label_folder, folder.split('/')[-1] + '.nrrd')
            lbl, options = nrrd.read(lbl_path)

            # fix swap axis
            lbl = np.swapaxes(lbl, 0, 1)

            # ================================ 
            # https://wiki.cancerimagingarchive.net/display/Public/NCI-ISBI+2013+Challenge+-+Automated+Segmentation+of+Prostate+Structures
            # A competitor reported an issue with case ProstateDx-01-0055, which has a dimension mismatch.
            # The segmentation has dimensions 400x400x23 whereas the DICOM image series have dimensions of 400x400x34.
            # We checked the case and indeed the dimensions seem to not correspond on Z (23 vs 34); however, the labels are properly spatially placed.
            # We don't currently see a problem with using the case. 
            # ================================ 
            if patname == 'ProstateDx-01-0055':
                lbl_tmp = np.zeros(shape = img.shape, dtype = lbl.dtype)
                lbl_tmp[:, :, :lbl.shape[2]] = lbl
                lbl = lbl_tmp
            
            # ================================
            # save as nifti, this sets the affine transformation as an identity matrix
            # ================================    
            utils.save_nii(img_path = nifti_img_path + '_lbl.nii.gz', data = lbl, affine = np.eye(4))
            
            nx_list[train_test].append(lbl.shape[0])
            ny_list[train_test].append(lbl.shape[1])
            nz_list[train_test].append(lbl.shape[2])

            print('lbl.shape')
            print(lbl.shape)
            print('img.shape')
            print(img.shape)

            ### PROCESSING LOOP FOR SLICE-BY-SLICE 2D DATA ###################
            scale_vector = [pixel_size[0] / target_resolution[0],
                            pixel_size[1] / target_resolution[1]]

            for zz in range(img.shape[2]):

                slice_img = np.squeeze(img[:, :, zz])
                slice_rescaled = transform.rescale(slice_img,
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

                slice_cropped = utils.crop_or_pad_slice_to_size(slice_rescaled, nx, ny)
                lbl_cropped = utils.crop_or_pad_slice_to_size(lbl_rescaled, nx, ny)

                img_list[train_test].append(slice_cropped)
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
                                sub_dataset = 'RUNMC', # RUNMC / BMC
                                cv_fold_num = 1):

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    data_file_name = 'data_2d_size_%s_res_%s_cv_fold_%d_%s.hdf5' % (size_str, res_str, cv_fold_num, sub_dataset)

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
# function to read a single subjects image and labels without any pre-processing
# TODO this function is still to be modified after the changes which make the rest of the functions in this file work for both subdatasets (RUNMC and BMC).
# ===============================================================
def load_without_size_preprocessing(input_folder,
                                    cv_fold_num,
                                    train_test,
                                    idx):
    
    # ===============================
    # read all the patient folders from the base input folder
    # ===============================
    image_folder = os.path.join(input_folder, 'Prostate-3T')
    label_folder = os.path.join(input_folder, 'NCI_ISBI_Challenge-Prostate3T_Training_Segmentations')
    folder_list = get_patient_folders(image_folder,
                                      folder_base='Prostate3T-01',
                                      cv_fold_number = cv_fold_num)
    folder = folder_list[train_test][idx]

    # ==================
    # make a list of all dcm images for this subject
    # ==================                        
    lstFilesDCM = []  # create an empty list
    for dirName, subdirList, fileList in os.walk(folder):
        for filename in fileList:
            if ".dcm" in filename.lower():  # check whether the file's DICOM
                lstFilesDCM.append(os.path.join(dirName, filename))
                
    # ==================
    # read bias corrected image
    # ==================
    nifti_img_path = lstFilesDCM[0][:lstFilesDCM[0].rfind('/')+1]
    image = utils.load_nii(img_path = nifti_img_path + 'img_n4.nii.gz')[0]

    # ============
    # normalize the image to be between 0 and 1
    # ============
    image = utils.normalise_image(image, norm_type='div_by_max')

    # ==================
    # read the label file
    # ==================        
    label = utils.load_nii(img_path = nifti_img_path + 'lbl.nii.gz')[0]
    
    return image, label

# ===============================================================
# ===============================================================
if __name__ == '__main__':
    input_folder = sys_config.orig_data_root_nci
    preprocessing_folder = sys_config.preproc_folder_nci

    data_nci = load_and_maybe_process_data(input_folder,
                                           preprocessing_folder,
                                           (256, 256),
                                           (0.625, 0.625),
                                           force_overwrite=True,
                                           sub_dataset = 'RUNMC',
                                           cv_fold_num = 1)