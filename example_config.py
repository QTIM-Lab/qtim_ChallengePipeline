import os

def default_config(config=None):

    """ Put default behaviors here """

    if config is None:
        config = dict()

    # Data will be compressed in hdf5 format at these filepaths.
    config["train_dir"] = ''
    config["validation_dir"] = None
    config["test_dir"] = ''

    # Data will be saved to these hdf5 files.
    config["hdf5_train"] = './hdf5_data/isles_train.hdf5'
    config["hdf5_validation"] = ''
    config["hdf5_test"] = './hdf5_data/isles_test_val.hdf5'

    # Overwrite settings.
    config["overwrite_trainval_data"] = True
    config['overwrite_test_data'] = True
    config["overwrite_model"] = True
    config["overwrite_training"] = True
    config["overwrite_prediction"] = True

    # Patch Information
    config['patch_shape'] = (24, 24, 4)
    config['training_patches_multiplier'] = 120
    config['validation_patches_multiplier'] = 10 

    # Perpetual Patch Information
    config["overwrite_masks"] = False
    config['perpetual_patches'] = False
    config['training_patch_multiplier'] = 50
    config['validation_patch_multiplier'] = 10
    config['roi_ratio'] = .9
    config['end_roi_ratio'] = .3
    config['roi_ratio_decline_steps'] = 50

    # Modalities. Always make input_groundtruth as list.
    config["training_modality_dict"] = {'input_modalities': ['ADC_pp.nii.gz', 'MTT_pp.nii.gz', 'rCBF_pp.nii.gz', 'rCBV_pp.nii.gz', 'TMax_pp.nii.gz', 'TPP_pp.nii.gz'],
                                        'ground_truth': ['groundtruth-label_raw.nii.gz']}
    config["test_modality_dict"] = {'input_modalities': ['ADC_pp.nii.gz', 'MTT_pp.nii.gz', 'rCBF_pp.nii.gz', 'rCBV_pp.nii.gz', 'TMax_pp.nii.gz', 'TPP_pp.nii.gz']}

    # Path to save model.
    config["model_file"] = "./model_files/isles_new.h5"

    # Brain Mask Options
    config["brain_mask_dir"] = './roi_masks/brain_masks/'
    config["roi_mask_dir"] = './roi_masks/roi_masks/'

    # Model parameters
    config["downsize_filters_factor"] = 1
    config["decay_learning_rate_every_x_epochs"] = 20
    config["initial_learning_rate"] = 0.00001
    config["learning_rate_drop"] = 0.9
    config["n_epochs"] = 100
    config["regression"] = False

    # Model training parameters
    config["training_batch_size"] = 50
    config["validation_batch_size"] = 50

    # Model testing parameters. More than needed, most likely.
    config['predict_with_hdf5'] = False
    config['predictions_folder'] = None
    config['predictions_name'] = 'infarct_prediction'
    config['predictions_input'] = 'input_modalities'
    config['predictions_groundtruth'] = None
    config['predictions_replace_existing'] = False
    config['prediction_output_num'] = 1
    config['prediction_repetitions'] = 16

    # Threshold Functions
    def background_patch(patch):
        return float((patch['input_modalities'] == 0).sum()) / patch['input_modalities'].size == 1

    def brain_patch(patch):
        return float((patch['input_modalities'] != 0).sum()) / patch['input_modalities'].size > .01 and float((patch['ground_truth'] == 1).sum()) / patch['ground_truth'].size < .01

    def roi_patch(patch):
        return float((patch['ground_truth'] == 1).sum()) / patch['ground_truth'].size > .01


    config["patch_extraction_conditions"] = [[background_patch, .02], [brain_patch, .199], [roi_patch, .8]]

    return config

def test_config(config=None):

    if config is None:
        config = default_config()

    config["overwrite_trainval_data"] = False
    config['overwrite_test_data'] = True
    config["overwrite_model"] = False
    config["overwrite_training"] = False
    config["overwrite_prediction"] = True

    return config

def predict_config(config=None):

    if config is None:
        config = default_config()

    config["overwrite_trainval_data"] = False
    config['overwrite_test_data'] = False
    config["overwrite_model"] = False
    config["overwrite_training"] = False
    config["overwrite_prediction"] = True

    return config

def train_config(config=None):

    if config is None:
        config = default_config()

    config["overwrite_trainval_data"] = False
    config['overwrite_test_data'] = False
    config["overwrite_model"] = True
    config["overwrite_training"] = True
    config["overwrite_prediction"] = False

    return config

def train_data_config(config=None):

    if config is None:
        config = default_config()

    config["overwrite_trainval_data"] = True
    config['overwrite_test_data'] = False
    config["overwrite_model"] = False
    config["overwrite_training"] = False
    config["overwrite_prediction"] = False

    return config

if __name__ == '__main__':
    pass