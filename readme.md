#### TTA Steps

1. Train a model using labelled data from the source domain by running train_i2l.py.
    This will train both $N_\phi$ and $S_\theta$.
	(Set hyper-parameters in experiments/i2l.py)

| Hyperparameters        			    | Value  | 
| :------------:                  | :---:  | 
| Training Dataset        			  | RUNMC / BMC / UCL / HK / BIDMC / USZ      | 
| Run number        						  | 1      | 
| Learning rate       						| 0.001  | 
| Batch size        						  | 16     | 
| Data Augmentation Ratio   			| 0.25   | 

	Learned model weights will be saved in:
	log_dir/tr<training_dataset>_r<run_number>/i2i2l/models/

2. Save 1D Gaussians for all channels of all layers for all training subjects, by running the file save_1d_gaussians.py. Following hyperparameters are relevant:

| Hyperparameters        				  | Value  | 
| :------------:                  | :---:  | 
| args.feature_subsampling_factor | 16 (features are subsampled before computing PDFs. Done here to be consistent with the KDE procedure, where subsampling is necessitated by memory issues.)      | 

	Subject-wise Gaussian parameters will be saved in:
	log_dir/tr<training_dataset>_r<run_number>/i2i2l/onedpdfs/

#### Brief description of required code files
| File        | Description  | 
| :------------:             | :---:  |
| config/params.py           |        | 
| config/system_paths.py     |        | 
| data/data_nci.py           |        | 
| data/data_pirad_erc.py     |        | 
| data/data_promise.py       |        | 
| experiments/i2l.py         |        | 
| tfwrapper/layers.py        |        | 
| tfwrapper/losses.py        |        | 
| evaluate.py                |        | 
| model_zoo.py               |        | 
| model.py                   |        | 
| train_i2l.py               |        | 
| utils_data.py              |        | 
| utils_kde.py               |        | 
| utils_vis.py               |        | 
| utils.py                   |        | 
| save_1d_gaussians.py       | saves 1D Gaussian parameters for all channels of all layers for all training subjects       | 
| save_1d_kdes.py            |        | 
| save_pca_kdes.py           |        | 