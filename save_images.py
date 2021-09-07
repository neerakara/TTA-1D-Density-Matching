import config.system_paths as sys_config
import data.data_hcp as data_hcp
import data.data_abide as data_abide
import data.data_nci as data_nci
import data.data_promise as data_promise
import data.data_pirad_erc as data_pirad_erc
import data.data_mnms as data_mnms
import data.data_wmh as data_wmh
import numpy as np
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt 
# import utils_vis

anatomy = 'brain' # prostate / cardiac / wmh / brain
dataset_name = 'HCPT1' # BMC, RUNMC, UCL, HK, BIDMC, USZ | UHE, CSF, HVHD | UMC, NUHS | HCPT1, CALTECH
image_size = (256, 256)
if anatomy == 'prostate':
    target_resolution = (0.625, 0.625)
elif anatomy == 'cardiac':
    target_resolution = (1.33, 1.33)
elif anatomy == 'wmh':
    target_resolution = (1.0, 1.0)
elif anatomy == 'brain':
    target_resolution = (0.7, 0.7)

if anatomy == 'brain':
    if dataset_name == 'CALTECH':
        data_brain = data_abide.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_abide,
                                                            preprocessing_folder = sys_config.preproc_folder_abide,
                                                            site_name = 'CALTECH',
                                                            idx_start = 16,
                                                            idx_end = 36,             
                                                            protocol = 'T1',
                                                            size = image_size,
                                                            depth = 256,
                                                            target_resolution = target_resolution)
    elif dataset_name == 'HCPT1':
        data_brain = data_hcp.load_and_maybe_process_data(input_folder = sys_config.orig_data_root_hcp,
                                                          preprocessing_folder = sys_config.preproc_folder_hcp,
                                                          idx_start = 50,
                                                          idx_end = 70,           
                                                          protocol = 'T1',
                                                          size = image_size,
                                                          depth = 256,
                                                          target_resolution = target_resolution)
    images = data_brain['images']
    labels = data_brain['labels']
    orig_data_siz_z = data_brain['nz'][:]
    patnames = data_brain['patnames']

else:
    if dataset_name in ['BMC', 'RUNMC']:
        data = data_nci.load_and_maybe_process_data(sys_config.orig_data_root_nci,
                                                    sys_config.preproc_folder_nci,
                                                    image_size,
                                                    target_resolution,
                                                    force_overwrite=False,
                                                    sub_dataset = dataset_name,
                                                    cv_fold_num = 1)

    elif dataset_name in ['UCL', 'BIDMC', 'HK']:
        data = data_promise.load_and_maybe_process_data(sys_config.orig_data_root_promise,
                                                        sys_config.preproc_folder_promise,
                                                        image_size,
                                                        target_resolution,
                                                        force_overwrite=False,
                                                        sub_dataset = dataset_name,
                                                        cv_fold_num = 1)

    elif dataset_name in ['USZ']:
        idx_start = 0 # test images
        idx_end = 20
        data = data_pirad_erc.load_data(sys_config.orig_data_root_pirad_erc,
                                        sys_config.preproc_folder_pirad_erc,
                                        idx_start = idx_start,
                                        idx_end = idx_end,
                                        size = image_size,
                                        target_resolution = target_resolution,
                                        labeller = 'ek')

    elif dataset_name in ['UHE', 'CSF', 'HVHD']:
        data = data_mnms.load_and_maybe_process_data(sys_config.orig_data_root_mnms,
                                                    sys_config.preproc_folder_mnms,
                                                    image_size,
                                                    target_resolution,
                                                    force_overwrite=False,
                                                    sub_dataset = dataset_name)

    elif dataset_name in ['UMC', 'NUHS']:
        data = data_wmh.load_and_maybe_process_data(sys_config.orig_data_root_wmh,
                                                    sys_config.preproc_folder_wmh,
                                                    image_size,
                                                    target_resolution,
                                                    force_overwrite=False,
                                                    sub_dataset = dataset_name,
                                                    cv_fold_number = 1,
                                                    protocol = 'FLAIR')
    images = data['images_test']
    labels = data['labels_test']
    patnames = data['patnames_test']
    orig_data_siz_z = data['nz_test'][:]

# extract one test image volume
for sub_num in range(10):# (orig_data_siz_z.shape[0]):
    subject_id_start_slice = np.sum(orig_data_siz_z[:sub_num])
    subject_id_end_slice = np.sum(orig_data_siz_z[:sub_num+1])
    image = images[subject_id_start_slice:subject_id_end_slice,:,:]  
    label = labels[subject_id_start_slice:subject_id_end_slice,:,:]  

    patname = str(patnames[sub_num])[2:-1]
    print(patname)
    print(image.shape)
    print(label.shape)

    image2d = image[image.shape[0]//2+20, :, :]
    label2d = label[image.shape[0]//2+20, :, :]

    # make all FG labels the same
    if anatomy == 'prostate':
        label[label!=0] = 1
    if anatomy == 'wmh':
        image2d = np.rot90(image2d, k=-1)
        label2d = np.rot90(label2d, k=-1)
    elif anatomy == 'brain':
        image2d = np.rot90(image2d, k=1)
        label2d = np.rot90(label2d, k=1)

    savepath_base = '/cluster/home/nkarani/projects/dg_seg/methods/tta_abn/v1/figures/data_vis/' + anatomy + '/'
    
    plt.figure(figsize=[20,20])                        
    plt.imshow(image2d, cmap='gray')
    plt.axis('off')
    plt.savefig(savepath_base + dataset_name + '_' + patname + '_image.png', bbox_inches='tight', pad_inches = 0, dpi=100)
    plt.close()

    plt.figure(figsize=[20,20])                        
    plt.imshow(label2d, cmap='gray')
    plt.axis('off')
    plt.savefig(savepath_base + dataset_name + '_' + patname + '_label.png', bbox_inches='tight', pad_inches = 0, dpi=100)
    plt.close()