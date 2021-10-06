import os
import numpy as np
import logging
import gc
import h5py
from skimage import transform
import glob
import zipfile, re
import utils, utils_vis
from skimage.transform import rescale
import config.system_paths as sys_config

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')

# Maximum number of data points that can be in memory at any time
MAX_WRITE_BUFFER = 5

# The raw data is in '/usr/bmicnas01/data-biwi-01/bmicdatasets-originals/Originals/HCP/3T_Structurals_Preprocessed/'
# Here, each subject has its own zip file.
# This data was extracted and copied to Euler, such that each subject has its own folder in '/cluster/work/cvl/nkarani/data/preprocessed/segmentation/HCP/'
# The unzipping process was done in the order returned by glob.
# Here is the list in which the subjects were returned by glob.
PATIENT_IDS_RETURNED_BY_GLOB = [100206, 100307, 100408, 100610, 101006, 101107,
                                101309, 101410, 101915, 102008, 102109, 102311,
                                102513, 102614, 102715, 102816, 103010, 103111, 
                                103212, 103414, 103515, 103818, 104012, 104416, 
                                104820, 105014, 105115, 105216, 105620, 105923,
                                106016, 106319, 106521, 106824, 107018, 107220, 
                                107321, 107422, 107725, 108020, 108121, 108222, 
                                108323, 108525, 108828, 109123, 109325, 109830,
                                110007, 110411]

# ===============================================================
# ===============================================================
def prepare_data(input_folder,
                 output_file,
                 idx_start,
                 idx_end,
                 protocol,
                 size,
                 depth,
                 target_resolution,
                 preprocessing_folder):

    # =======================
    # =======================
    hdf5_file = h5py.File(output_file, "w")

    # ===============================
    # Create an empty dataset for labels
    # ===============================
    data = {}
    num_subjects = idx_end - idx_start
    data['labels'] = hdf5_file.create_dataset("labels", [num_subjects] + list(size), dtype=np.uint8)
    
    # ===============================
    # initialize lists
    # ===============================        
    label_list = []
    nx_list = []
    ny_list = []
    nz_list = []
    px_list = []
    py_list = []
    pz_list = []
    pat_names_list = []
    
    # ===============================        
    # initiate counter
    # ===============================        
    patient_counter = 0
    
    # ===============================
    # iterate through the requested indices
    # ===============================
    for idx in range(idx_start, idx_end):
        
        # ==================
        # get file path
        # ==================
        patient_name = str(PATIENT_IDS_RETURNED_BY_GLOB[idx])
        label_path = input_folder + patient_name + '/T1w/aparc+aseg.nii.gz'
    
        # ==================
        # read the label file
        # ==================        
        label, _, label_hdr = utils.load_nii(label_path)        
        label = np.swapaxes(label, 1, 2) # swap axes 1 and 2 -> this allows appending along axis 2, as in other datasets
        label = utils.group_segmentation_classes(label) # group the segmentation classes as required
        logging.info("original label shape: " + str(label.shape))

        # ==================
        # collect some header info.
        # ==================
        px_list.append(float(label_hdr.get_zooms()[0]))
        py_list.append(float(label_hdr.get_zooms()[2])) # since axes 1 and 2 have been swapped
        pz_list.append(float(label_hdr.get_zooms()[1]))
        nx_list.append(label.shape[0]) 
        ny_list.append(label.shape[2]) # since axes 1 and 2 have been swapped
        nz_list.append(label.shape[1])
        pat_names_list.append(patient_name)
        
        # ==================
        # crop volume along z axis (as there are several zeros towards the ends)
        # ==================
        label = utils.crop_or_pad_volume_to_size_along_z(label, depth)     
        logging.info("label shape after initial crop: " + str(label.shape))
        
        logging.info("Initial resolution: " + str(label_hdr.get_zooms()))

        # ======================================================  
        # rescale, crop / pad to make all images of the required size and resolution
        # ======================================================
        scale_vector = [label_hdr.get_zooms()[0] / target_resolution[2],
                        label_hdr.get_zooms()[2] / target_resolution[1],
                        label_hdr.get_zooms()[1] / target_resolution[0]]
        
        # This way of downsampling leads to artifacts with scikit-lear 0.17.2. It was working fine with 0.14.0
        # label_onehot = utils.make_onehot(label, nlabels=15)
        # label_onehot_rescaled = transform.rescale(label_onehot,
        #                                           scale_vector,
        #                                           order=1,
        #                                           preserve_range=True,
        #                                           multichannel=True,
        #                                           mode='constant',
        #                                           anti_aliasing=True)
        # label_rescaled = np.argmax(label_onehot_rescaled, axis=-1)

        # This works fine.
        label_rescaled = transform.rescale(label,
                                           scale_vector,
                                           order=0,
                                           preserve_range=True,
                                           multichannel=False,
                                           mode='constant',
                                           anti_aliasing=False)

        logging.info("Shape after rescaling: " + str(label_rescaled.shape))
        
        # ==================================
        # go through each z slice, crop or pad to a constant size and then append the resized 
        # this will ensure that the axes get arranged in the same orientation as they were during the 2d preprocessing
        # ==================================
        label_rescaled_cropped = []
        for zz in range(label_rescaled.shape[2]):
            label_rescaled_cropped.append(utils.crop_or_pad_slice_to_size(label_rescaled[:,:,zz], size[1], size[2]))
        label_rescaled_cropped = np.array(label_rescaled_cropped)
        logging.info("Shape after cropping: " + str(label_rescaled_cropped.shape))

        # ============   
        # append to list
        # ============   
        label_list.append(label_rescaled_cropped)

        # ============   
        # write to file
        # ============   
        _write_range_to_hdf5(data,
                             label_list,
                             patient_counter,
                             patient_counter+1)
        
        _release_tmp_memory(label_list)
        
        # update counter
        patient_counter += 1

    # Write the small datasets
    hdf5_file.create_dataset('nx', data=np.asarray(nx_list, dtype=np.uint16))
    hdf5_file.create_dataset('ny', data=np.asarray(ny_list, dtype=np.uint16))
    hdf5_file.create_dataset('nz', data=np.asarray(nz_list, dtype=np.uint16))
    hdf5_file.create_dataset('px', data=np.asarray(px_list, dtype=np.float32))
    hdf5_file.create_dataset('py', data=np.asarray(py_list, dtype=np.float32))
    hdf5_file.create_dataset('pz', data=np.asarray(pz_list, dtype=np.float32))
    hdf5_file.create_dataset('patnames', data=np.asarray(pat_names_list, dtype="S10"))
    
    # After test train loop:
    hdf5_file.close()

# ===============================================================
# Helper function to write a range of data to the hdf5 datasets
# ===============================================================
def _write_range_to_hdf5(hdf5_data,
                         lbl_list,
                         counter_from,
                         counter_to):

    logging.info('Writing data from %d to %d' % (counter_from, counter_to))
    lbl_arr = np.asarray(lbl_list, dtype=np.uint8)
    hdf5_data['labels'][counter_from : counter_to, ...] = lbl_arr

# ===============================================================
# Helper function to reset the tmp lists and free the memory
# ===============================================================
def _release_tmp_memory(lbl_list):

    lbl_list.clear()
    gc.collect()
    
# ===============================================================
# ===============================================================
def load_and_maybe_process_data(input_folder,
                                preprocessing_folder,
                                idx_start,
                                idx_end,
                                protocol,
                                size,
                                depth,
                                target_resolution,
                                force_overwrite=False):

    size_str = '_'.join([str(i) for i in size])
    res_str = '_'.join([str(i) for i in target_resolution])

    data_file_name = 'data_%s_3d_size_%s_depth_%d_res_%s_from_%d_to_%d.hdf5' % (protocol, size_str, depth, res_str, idx_start, idx_end)
    data_file_path = os.path.join(preprocessing_folder, data_file_name)

    utils.makefolder(preprocessing_folder)

    if not os.path.exists(data_file_path) or force_overwrite:
        logging.info('This configuration of mode, size and target resolution has not yet been preprocessed')
        logging.info('Preprocessing now!')
        prepare_data(input_folder,
                     data_file_path,
                     idx_start,
                     idx_end,
                     protocol,
                     size,
                     depth,
                     target_resolution,
                     preprocessing_folder)
    else:
        logging.info('Already preprocessed this configuration. Loading now!')

    return h5py.File(data_file_path, 'r')

# ===============================================================
# ===============================================================
if __name__ == '__main__':
    
    input_folder = sys_config.orig_data_root_hcp
    preprocessing_folder = sys_config.preproc_folder_hcp

    data_hcp = load_and_maybe_process_data(input_folder,
                                           preprocessing_folder,
                                           idx_start = 20,
                                           idx_end = 25,
                                           protocol = 'T1',
                                           size = (64, 256, 256),
                                           depth = 256, # initial crop out length before downsampling
                                           target_resolution = (2.8, 0.7, 0.7),
                                           force_overwrite=False)

    print(data_hcp['labels'].shape)
    
    gttr = data_hcp['labels']
    savepath = '/cluster/work/cvl/nkarani/projects/dg_seg/methods/tta_abn/v2/log_dir/trHCPT1_cv1_r1/i2i2l/tta/DAE/r1/'
    visualize = True
    if visualize:
        for subject_num in range(gttr.shape[0]):
            utils_vis.save_samples_downsampled(gttr[subject_num, ...], savepath = savepath + 'tr_image_' + str(subject_num+1) + '.png')

    data_hcp.close()