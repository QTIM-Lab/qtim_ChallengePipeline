import tables
import os
import math

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from functools import partial
from shutil import rmtree

from keras.callbacks import ModelCheckpoint, CSVLogger, Callback, LearningRateScheduler
from keras.utils import plot_model
from keras.backend import clear_session

import config_files.BRATS_2017.edema_config as edema_config
import config_files.BRATS_2017.tumor1_config as tumor1_config
import config_files.BRATS_2017.tumor2_config as tumor2_config
import config_files.BRATS_2017.nonenhancing1_config as nonenhancing1_config
import config_files.BRATS_2017.nonenhancing2_config as nonenhancing2_config
import config_files.BRATS_2017.reconciliation_config as reconciliation_config
import config_files.BRATS_2017.isles_config as isles_config

from model import n_net_3d, u_net_3d, split_u_net_3d, w_net_3d, load_old_model, vox_net, parellel_unet_3d
from load_data import DataCollection
from data_generator import get_data_generator
from data_utils import pickle_dump, pickle_load
from predict import model_predict_patches
from augment import *
from file_util import split_folder

def learning_pipeline(overwrite=False, delete=False, config=None, parameters=None):

    # append_prefix_to_config(config, ["hdf5_train", "hdf5_validation", "hdf5_test"], 'downsample_')
    # append_prefix_to_config(config, ["model_file"], 'downsample_')    
    # config['predictions_name'] = 'downsample_edema_prediction'
    config['patch_shape'] = (24, 24, 4)
    config['training_patches_multiplier'] = 50
    config['validation_patches_multiplier'] = 10

    # config['overwrite_trainval_data'] = True
    # append_prefix_to_config(config, ["model_file"], 'downsize_4_regression_')
    # config["downsize_filters_factor"] = 1
    # config["initial_learning_rate"] = 0.00001
    # config["regression"] = True
    config['predictions_replace_existing'] = True
    config["n_epochs"] = 200
    config["batch_size"] = 1
    config["image_shape"] = None

    update_config(config=config, parameters=parameters)
    create_directories(delete=delete, config=config)

    modality_dict = config['training_modality_dict']
    validation_files = []

    # Load training and validation data.
    if config['overwrite_trainval_data'] or not os.path.exists(os.path.abspath(config["hdf5_train"])):

        print 'WRITING DATA', '\n'

        # Find Data
        training_data_collection = DataCollection(config['train_dir'], modality_dict)
        training_data_collection.fill_data_groups()

        flip_augmentation = Flip_Rotate_2D(flip=True, rotate=False)
        flip_augmentation_group = AugmentationGroup({'input_modalities': flip_augmentation, 'ground_truth': flip_augmentation}, multiplier=2)

        # Training - with patch augmentation
        patch_extraction_augmentation = ExtractPatches(config['patch_shape'], config['patch_extraction_conditions'])
        training_patch_augmentation_group = AugmentationGroup({'input_modalities': patch_extraction_augmentation, 'ground_truth': patch_extraction_augmentation}, multiplier=config['training_patches_multiplier'])

        training_data_collection.append_augmentation(training_patch_augmentation_group)
        training_data_collection.append_augmentation(flip_augmentation_group)

        training_data_collection.write_data_to_file(output_filepath = config['hdf5_train'])

        # Validation - with patch augmentation
        validation_data_collection = DataCollection(config['validation_dir'], modality_dict)
        validation_data_collection.fill_data_groups()

        validation_patch_augmentation_group = AugmentationGroup({'input_modalities': patch_extraction_augmentation, 'ground_truth': patch_extraction_augmentation}, multiplier=config['validation_patches_multiplier'])
        validation_data_collection.append_augmentation(validation_patch_augmentation_group)
        validation_data_collection.append_augmentation(flip_augmentation_group)

        validation_data_collection.write_data_to_file(output_filepath = config['hdf5_validation'])

    # Create a new model if necessary. Preferably, load an existing one.
    if not config["overwrite_model"] and os.path.exists(config["model_file"]):
        model = load_old_model(config["model_file"])
    else:
        model = split_u_net_3d(input_shape=(len(modality_dict['input_modalities']),) + config['patch_shape'], output_shape=(len(modality_dict['ground_truth']),) + config['patch_shape'], downsize_filters_factor=config['downsize_filters_factor'], initial_learning_rate=config['initial_learning_rate'], regression=config['regression'], num_outputs=(len(modality_dict['ground_truth'])))

    plot_model(model, to_file='model4.png', show_shapes=True)

    # Create data generators and train the model.
    if config["overwrite_training"]:
        
        # rotation_augmentation = Flip_Rotate_2D(config['patch_shape'], config['patch_extraction_conditions'])
        # rotation_augmentation_group = AugmentationGroup({'input_modalities': rotation_augmentation, 'ground_truth': rotation_augmentation,}, multiplier=1)

        # noise_augmentation = GaussianNoise(config['patch_shape'], config['patch_extraction_conditions'])
        # copy_augmentation = Copy()
        # noise_augmentation_group = AugmentationGroup({'input_modalities': patch_extraction_augmentation, 'ground_truth': copy_augmentation}, multiplier=1)
        # noise_augmentation_group.append_augmentation(patch_augmentation_group)


        # Get training and validation generators, either split randomly from the training data or from separate hdf5 files.
        if os.path.exists(os.path.abspath(config["hdf5_validation"])):
            open_validation_hdf5 = tables.open_file(config["hdf5_validation"], "r")
            validation_generator, num_validation_steps = get_data_generator(open_validation_hdf5, batch_size=1, data_labels = ['input_modalities', 'ground_truth'])
        else:
            open_validation_hdf5 = []

        open_train_hdf5 = tables.open_file(config["hdf5_train"], "r")
        train_generator, num_train_steps = get_data_generator(open_train_hdf5, batch_size=config["batch_size"], data_labels = ['input_modalities', 'ground_truth'])

        # Train model.. TODO account for no validation
        train_model(model=model, model_file=config["model_file"], training_generator=train_generator, validation_generator=validation_generator, steps_per_epoch=num_train_steps, validation_steps=num_validation_steps, initial_learning_rate=config["initial_learning_rate"], learning_rate_drop=config["learning_rate_drop"], learning_rate_epochs=config["decay_learning_rate_every_x_epochs"], n_epochs=config["n_epochs"])

        # Close training and validation files, no longer needed.
        open_train_hdf5.close()
        if validation_files:
            open_validation_hdf5.close()

    # Load testing data
    if config['overwrite_test_data'] or not os.path.exists(os.path.abspath(config["hdf5_test"])):
        
        modality_dict = config['test_modality_dict']

        testing_data_collection = DataCollection(config['test_dir'], modality_dict)
        testing_data_collection.fill_data_groups()

        # testing_data_collection.write_data_to_file(output_filepath = config['hdf5_test'])
        testing_data_collection.write_data_to_list()

    # Run prediction step.
    if config['overwrite_prediction']:
        open_test_hdf5 = tables.open_file(config["hdf5_test"], "r")
        model_predict_patches(output_directory=config['predictions_folder'], output_name=config['predictions_name'], input_data_label=config['predictions_input'], output_data_label=config['predictions_groundtruth'], model=model, data_file=open_test_hdf5, patch_shape=config['patch_shape'], output_shape=config['image_shape'], replace_existing=config['predictions_replace_existing'])

def create_directories(delete=False, config=None):

    # Create required directories
    for directory in [config['model_file'], config['hdf5_train'], config['hdf5_test'], config['hdf5_validation'], config['predictions_folder']]:
        if directory is not None:
            directory = os.path.abspath(directory)
            if not os.path.isdir(directory):
                directory = os.path.dirname(directory)
            if delete:
                rmtree(directory)
            if not os.path.exists(directory):
                os.makedirs(directory)

    clear_session()

def update_config(config, parameters):

    if parameters is None:
        return

    for key in parameters.keys():
        config[key] = parameters[key]

def append_prefix_to_config(config, keys, prefix):

    for key in keys:
        config[key] = '/'.join(str.split(config[key], '/')[0:-1]) + '/' + prefix + str.split(config[key], '/')[-1]

def train_model(model, model_file, training_generator, validation_generator, steps_per_epoch, validation_steps, initial_learning_rate, learning_rate_drop, learning_rate_epochs, n_epochs):

    model.fit_generator(generator=training_generator, steps_per_epoch=steps_per_epoch, epochs=n_epochs, validation_data=validation_generator, validation_steps=validation_steps, pickle_safe=True, callbacks=get_callbacks(model_file, initial_learning_rate=initial_learning_rate, learning_rate_drop=learning_rate_drop,learning_rate_epochs=learning_rate_epochs))

    model.save(model_file)

""" The following three functions/classes are mysterious to me.
"""

def step_decay(epoch, initial_lrate, drop, epochs_drop):
    return initial_lrate * math.pow(drop, math.floor((1+epoch)/float(epochs_drop)))

class SaveLossHistory(Callback):
    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        pickle_dump(self.losses, "loss_history.pkl")

def get_callbacks(model_file, initial_learning_rate, learning_rate_drop, learning_rate_epochs, logging_dir="."):

    """ Currently do not understand callbacks.
    """

    model_checkpoint = ModelCheckpoint(model_file, save_best_only=True)
    logger = CSVLogger(os.path.join(logging_dir, "training.log"))
    history = SaveLossHistory()
    scheduler = LearningRateScheduler(partial(step_decay, initial_lrate=initial_learning_rate, drop=learning_rate_drop, epochs_drop=learning_rate_epochs))
    return [model_checkpoint, logger, history, scheduler]

if __name__ == '__main__':

    # learning_pipeline(config=isles_config.default_config(), overwrite=False)
    # learning_pipeline(config=isles_config.train_config(), overwrite=False)
    learning_pipeline(config=isles_config.test_config(), overwrite=False)

    # learning_pipeline(config=edema_config.train_config(), overwrite=False)
    # learning_pipeline(config=edema_config.predict_config(), overwrite=False)
    # learning_pipeline(config=tumor1_config.train_config(), overwrite=False)
    # learning_pipeline(config=tumor1_config.predict_config(), overwrite=False)
    # learning_pipeline(config=nonenhancing1_config.train_config(), overwrite=False)
    # learning_pipeline(config=nonenhancing1_config.test_config(), overwrite=False)
    # learning_pipeline(config=tumor2_config.train_config(), overwrite=False)
    # learning_pipeline(config=tumor2_config.test_config(), overwrite=False)
    # learning_pipeline(config=nonenhancing2_config.train_config(), overwrite=False)
    # learning_pipeline(config=nonenhancing2_config.test_config(), overwrite=False)


    # learning_pipeline(config=edema_config.test_config(), overwrite=False,  parameters={"test_dir": '/mnt/jk489/sharedfolder/BRATS2017/Train'})
    # learning_pipeline(config=tumor1_config.test_config(), overwrite=False, parameters={"test_dir": '/mnt/jk489/sharedfolder/BRATS2017/Train'})
    # learning_pipeline(config=nonenhancing1_config.test_config(), overwrite=False,  parameters={"test_dir": '/mnt/jk489/sharedfolder/BRATS2017/Train'})    
    # learning_pipeline(config=tumor2_config.test_config(), overwrite=False, parameters={"test_dir": '/mnt/jk489/sharedfolder/BRATS2017/Train'})
    # learning_pipeline(config=nonenhancing2_config.test_config(), overwrite=False,  parameters={"test_dir": '/mnt/jk489/sharedfolder/BRATS2017/Train'})

    # learning_pipeline(config=edema_config.test_config(), overwrite=False,  parameters={"test_dir": '/mnt/jk489/sharedfolder/BRATS2017/Validation'})
    # learning_pipeline(config=tumor1_config.test_config(), overwrite=False, parameters={"test_dir": '/mnt/jk489/sharedfolder/BRATS2017/Validation'})
    # learning_pipeline(config=nonenhancing1_config.test_config(), overwrite=False,  parameters={"test_dir": '/mnt/jk489/sharedfolder/BRATS2017/Validation'})    
    # learning_pipeline(config=tumor2_config.test_config(), overwrite=False, parameters={"test_dir": '/mnt/jk489/sharedfolder/BRATS2017/Validation'})
    # learning_pipeline(config=nonenhancing2_config.test_config(), overwrite=False,  parameters={"test_dir": '/mnt/jk489/sharedfolder/BRATS2017/Validation'})

    # split_folder('/mnt/jk489/sharedfolder/BRATS2017/Val', .2, ['/mnt/jk489/sharedfolder/BRATS2017/Val_Train', '/mnt/jk489/sharedfolder/BRATS2017/Val_Val'])
    # learning_pipeline(config=reconciliation_config.train_config(), overwrite=False)
    # learning_pipeline(config=reconciliation_config.test_config(), overwrite=False)