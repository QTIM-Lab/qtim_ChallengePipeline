
import os

import nibabel as nib
import numpy as np
import tables
from keras import backend as K
from model import load_old_model
from image_utils import save_numpy_2_nifti, nifti_2_numpy
from file_util import replace_suffix, nifti_splitext
from load_data import read_image_files

import multiprocessing
from functools import partial

from joblib import Parallel, delayed

K.set_learning_phase(1)

def model_predict_patches_hdf5(data_file, input_data_label, patch_shape, repetitions=24, test_batch_size=200, ground_truth_data_label=None, output_shape=None, model=None, model_file=None, output_directory=None, output_name=None, replace_existing=True, merge_labels=True):

    """ TODO: Make work for multiple inputs and outputs.
        TODO: Interact with data group interface
        TODO: Pass output filenames to hdf5 files.
    """

    # Create output directory. If not provided, output into original patient folder.
    if output_directory is not None:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    # Load model.
    if model is None and model_file is None:
        print 'Error. Please provide either a model object or a model filepath.'
    elif model is None:
        model = load_old_model(model_file)

    # TODO: Add check in case an object is passed in.
    # input_data_label_object = self.data_groups[input_data_label_group]

    # Preallocate Data
    data_list = getattr(data_file.root, input_data_label)

    casename_list = getattr(data_file.root, '_'.join([input_data_label + '_casenames']))
    casename_list = [np.array_str(np.squeeze(x)) for x in casename_list]

    affine_list = getattr(data_file.root, '_'.join([input_data_label + '_affines']))
    affine_list = [np.squeeze(x) for x in affine_list]

    total_case_num = data_list.shape[0]

    if ground_truth_data_label is not None:
        truth_list = getattr(data_file.root, ground_truth_data_label)

    for case_idx in xrange(total_case_num):

        print 'Working on image.. ', case_idx, 'in', total_case_num

        # Filename for output predictions. TODO: Make a more informative output for output_name == None
        if output_name == None:
            case_output_name = 'TESTCASE_' + str(case_idx).zfill(3) + '_PREDICT'
        else:
            case_output_name = output_name

        # Destination for predictions. If not in new folder, predict in the same folder as the original image.
        if output_directory is not None:
            output_filepath = os.path.join(output_directory, case_output_name + '.nii.gz')
        else:
            case_directory = casename_list[case_idx]
            output_filepath = os.path.join(case_directory, case_output_name + '.nii.gz')
            print os.path.basename(case_directory)

        print output_filepath
        # If prediction already exists, skip it. Useful if process is interrupted.
        if os.path.exists(output_filepath) and not replace_existing:
            continue

        # Get data from hdf5
        case_input_data = np.asarray([data_list[case_idx]])
        case_affine = affine_list[case_idx]

        # Get groundtruth if provided.
        if ground_truth_data_label is not None:
            case_groundtruth_data = np.asarray([truth_list[case_idx]])
        else:
            case_groundtruth_data = None

        # Get the shape of the output either from input data, groundtruth, or pre-specification.
        if ground_truth_data_label is None and output_shape is None:
            output_shape = list(case_input_data.shape)
            output_shape[1] = 1
            output_shape = tuple(output_shape)
        elif output_shape is None:
            output_shape = case_groundtruth_data.shape

        for activations in [40,10,62]:
            temp_filepath = replace_suffix(output_filepath, '', '_' + str(activations))
            print temp_filepath
            output_data = predict_patches_one_image(case_input_data, patch_shape, model, output_shape, repetitions=repetitions, model_batch_size=test_batch_size, activation=activations)

            save_prediction(output_data, temp_filepath, input_affine=case_affine, ground_truth=case_groundtruth_data)

    data_file.close()

def model_predict_patches_collection(data_collection, input_data_label, patch_shape, repetitions=10, test_batch_size=200, ground_truth_data_label=None, output_shape=None, model=None, model_file=None, output_directory=None, output_name=None, replace_existing=True, merge_labels=True):

    """ TODO: Make work for multiple inputs and outputs.
        TODO: Interact with data group interface
        TODO: Pass output filenames to hdf5 files.
    """

    # Create output directory. If not provided, output into original patient folder.
    if output_directory is not None:
        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

    # Load model.
    if model is None and model_file is None:
        print 'Error. Please provide either a model object or a model filepath.'
    elif model is None:
        model = load_old_model(model_file)

    # Filename for output predictions. TODO: Make a more informative output for output_name == None
    if output_name == None:
        case_output_name = 'TESTCASE_' + str(case_idx).zfill(3) + '_PREDICT'
    else:
        case_output_name = output_name

    case_list = data_collection.data_groups[input_data_label].cases
    input_data_list = data_collection.data_groups[input_data_label].data

    if ground_truth_data_label is not None:
        groundtruth_data_list = data_collection.data_groups[ground_truth_data_label].data
    total_case_num = len(case_list)

    for case_idx in xrange(total_case_num):

        case_folder = case_list[case_idx]
        input_data_filepaths = input_data_list[case_idx]
        if ground_truth_data_label is not None:
            groundtruth_data_filepaths = groundtruth_data_list[case_idx]

        # Destination for predictions. If not in new folder, predict in the same folder as the original image.
        if output_directory is not None:
            output_filepath = os.path.join(output_directory, case_output_name + '.nii.gz')
        else:
            output_filepath = os.path.join(case_folder, case_output_name + '.nii.gz')

        # If prediction already exists, skip it. Useful if process is interrupted.
        if os.path.exists(output_filepath) and not replace_existing:
            continue

        print case_folder
        print 'Working on image.. ', case_idx, 'in', total_case_num

        # Filename for output predictions. TODO: Make a more informative output for output_name == None
        if output_name == None:
            case_output_name = 'TESTCASE_' + str(case_idx).zfill(3) + '_PREDICT'
        else:
            case_output_name = output_name

        # If prediction already exists, skip it. Useful if process is interrupted.
        if os.path.exists(output_filepath) and not replace_existing:
            continue

        # Get data from hdf5
        case_input_data, case_affine = read_image_files(input_data_filepaths, return_affine=True)
        case_input_data = np.expand_dims(case_input_data, 0)

        # Get groundtruth if provided.
        if ground_truth_data_label is not None:
            case_groundtruth_data = np.expand_dims(read_image_files(groundtruth_data_filepaths), 0)
        else:
            case_groundtruth_data = None

        # Get the shape of the output either from input data, groundtruth, or pre-specification.
        # if ground_truth_data_label is None and output_shape is None:
        output_shape = list(case_input_data.shape)
        output_shape[1] = 1
        output_shape = tuple(output_shape)

        output_data = predict_patches_one_image(case_input_data, patch_shape, model, output_shape, repetitions=repetitions, model_batch_size=test_batch_size, layer_output=None)

        # save_prediction(output_data, temp_filepath, input_affine=case_affine, ground_truth=case_groundtruth_data)

        # output_data = write_prediction_to_npy(case_input_data, patch_shape, model, output_shape, repetitions=repetitions, model_batch_size=test_batch_size)

        save_prediction(output_data, output_filepath, input_affine=case_affine, ground_truth=case_groundtruth_data)


def predict_patches_one_image(input_data, patch_shape, model, output_shape, repetitions=16, model_batch_size=200, layer_output=1, activation=0):

    """ Presumes data is in the format (batch_size, channels, dims)
    """

    # Should we automatically determine output_shape?
    # output_shape = (output_shape[0], ) + (64, ) + output_shape[2:]
    output_data = np.zeros(output_shape)

    repetition_offsets = [np.linspace(0, patch_shape[x]-1, repetitions, dtype=int) for x in xrange(len(patch_shape))]
    for rep_idx in xrange(repetitions):

        print 'PREDICTION PATCH GRID REPETITION # ..', rep_idx

        offset_slice = [slice(min(repetition_offsets[axis][rep_idx], input_data.shape[axis+2]-patch_shape[axis]), None, 1) for axis in xrange(len(patch_shape))]
        offset_slice = [slice(None)]*2 + offset_slice
        repatched_image = np.zeros_like(output_data[offset_slice])

        corners_list = patchify_image(input_data[offset_slice], [input_data[offset_slice].shape[1]] + list(patch_shape))

        for corner_list_idx in xrange(0, len(corners_list), model_batch_size):

            corner_batch = corners_list[corner_list_idx:corner_list_idx+model_batch_size]
            input_patches = grab_patch(input_data[offset_slice], corners_list[corner_list_idx:corner_list_idx+model_batch_size], patch_shape)

            prediction = model.predict(input_patches)

            for corner_idx, corner in enumerate(corner_batch):
                insert_patch(repatched_image, prediction[corner_idx, ...], corner)

        if rep_idx == 0:
            output_data = np.copy(repatched_image)
        else:
            # Running Average
            output_data[offset_slice] = output_data[offset_slice] + (1.0 / (rep_idx)) * (repatched_image - output_data[offset_slice])

    return output_data

def create_hdf5_file(hdf5_filepath, shape):

    # Investigate hdf5 files.
    hdf5_file = tables.open_file(hdf5_filepath, mode='w')

    # Investigate this line.
    # Compression levels = complevel. No compression = 0
    # Compression library = Method of compresion.
    filters = tables.Filters(complevel=5, complib='blosc')

    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=num_cases)

    coordinate_storage = hdf5_file.create_earray(hdf5_file.root, '_'.join(['coordinates']), tables.Float32Atom(), shape=(0,3), filters=filters, expectedrows=num_cases)

    return hdf5_file, data_storage, coordinate_storage

def write_prediction_to_npy(input_data, patch_shape, model, output_shape, repetitions=16, model_batch_size=200, layer_output=14):

    if layer_output is not None:
        get_specific_layer_output = K.function([model.layers[0].input], [model.layers[layer_output].output])
        output_prediction_data = None
        output_coordinate_data = None

    input_data[:,:,-patch_shape[0]/2:,...] = 0
    input_data[:,:,:,-patch_shape[1]/2:,...] = 0
    input_data[:,:,:,:,-patch_shape[2]/2:] = 0
    input_data[:,:,:patch_shape[0]/2,...] = 0
    input_data[:,:,:,:patch_shape[1]/2,...] = 0
    input_data[:,:,:,:,:patch_shape[2]/2] = 0
    non_zero_corners = np.nonzero(input_data)
    print len(non_zero_corners)
    print non_zero_corners[0]
    non_zero_corners = np.array(non_zero_corners)[1:,:].T
    print non_zero_corners.shape
    np.random.shuffle(non_zero_corners)

    # if True:
    try:
        for corner_idx in xrange(0, non_zero_corners.shape[0], model_batch_size):

            corners_list = non_zero_corners[corner_idx:corner_idx+model_batch_size]
            print corners_list.shape

            # for corner in corners_list:
            #     print corner
            # print len(corners_list)
            # print np.max(np.array(corners_list), axis=0)

            input_patches = grab_patch(input_data, corners_list, patch_shape)

            print input_patches.shape
            prediction = get_specific_layer_output([input_patches])[0]
            midpoints = np.array([[corner[1] + patch_shape[0]/2, corner[2]+ patch_shape[1]/2, corner[3] + patch_shape[2]/2]  for corner in corners_list])
            if output_prediction_data is None:
                output_prediction_data = prediction
                output_coordinate_data = midpoints
            else:
                output_prediction_data = np.concatenate((output_prediction_data, prediction))
                output_coordinate_data = np.concatenate((
                    output_coordinate_data, midpoints))
            print output_prediction_data.shape
            print output_coordinate_data.shape
    except:
        pass

    np.save('coords_array.npy', output_coordinate_data)
    np.save('data_array.npy', output_prediction_data)

    return

def save_prediction(input_data, output_filepath, input_affine=None, ground_truth=None, stack_outputs=False, binarize_probability=.5):

    """ This is a function just for function's sake
        TODO: Parse out the most logical division of prediction functions.
    """

    # If no affine, create identity affine.
    if input_affine is None:
        input_affine = np.eye(4)

    output_shape = input_data.shape
    input_data = np.squeeze(input_data)

    # If output modalities is one, just save the output.
    if output_shape[1] == 1:
        binarized_output_data = threshold_binarize(threshold=binarize_probability, input_data=input_data)
        print 'SUM OF ALL PREDICTION VOXELS', np.sum(binarized_output_data)
        save_numpy_2_nifti(input_data, reference_affine=input_affine, output_filepath=replace_suffix(output_filepath, input_suffix='', output_suffix='-probability'))
        # save_numpy_2_nifti(binarized_output_data, reference_affine=input_affine, output_filepath=replace_suffix(output_filepath, input_suffix='', output_suffix='-label'))
        # if ground_truth is not None:
            # print 'DICE COEFFICIENT', calculate_prediction_dice(binarized_output_data, np.squeeze(ground_truth))
    
    # If multiple output modalities, either stack one on top of the other (e.g. output 3 over output 2 over output 1).
    # or output multiple volumes.
    else:
        if stack_outputs:
            merge_image = threshold_binarize(threshold=binarize_probability, input_data=input_data[0,...])
            print 'SUM OF ALL PREDICTION VOXELS, MODALITY 0', np.sum(merge_image)
            for modality_idx in xrange(1, output_shape[1]):
                print 'SUM OF ALL PREDICTION VOXELS, MODALITY',str(modality_idx), np.sum(input_data[modality_idx,...])
                merge_image[threshold_binarize(threshold=binarize_probability, input_data=input_data[modality_idx,...]) == 1] = modality_idx

            save_numpy_2_nifti(threshold_binarize(threshold=binarize_probability, input_data=input_data[modality,...]), reference_affine=input_affine, output_filepath=output_filepath)
    
        for modality in xrange(output_shape[1]):
            print 'SUM OF ALL PREDICTION VOXELS, MODALITY',str(modality), np.sum(input_data[modality,...])
            binarized_output_data = threshold_binarize(threshold=binarize_probability, input_data=input_data[modality,...])
            save_numpy_2_nifti(input_data[modality,...], reference_affine=input_affine, output_filepath=replace_suffix(output_filepath, input_suffix='', output_suffix='_' + str(modality) + '-probability'))
            save_numpy_2_nifti(binarized_output_data, reference_affine=input_affine, output_filepath=replace_suffix(output_filepath, input_suffix='', output_suffix='_' + str(modality) + '-label'))

    return

def patchify_image(input_data, patch_shape, offset=(0,0,0,0), batch_dim=True, return_patches=False, mask_value = 0):

    """ VERY wonky. Patchs an image of arbitrary dimension, but
        has some interesting assumptions built-in about batch sizes,
        channels, etc.

        TODO: Make this function able to iterate forward or backward.
    """

    corner = [0] * len(input_data.shape[1:])

    if return_patches:
        patch = grab_patch(input_data, corner, patch_shape)
        patch_list = [[corner[:], patch[:]]]
    else:
        patch_list = [corner[:]]

    finished = False

    while not finished:

        # Wonky, fix in grab patch.
        # print input_data.shape
        # print corner
        # print patch_shape
        patch = grab_patch(input_data, [corner], tuple(patch_shape[1:]))
        # print patch.shape
        # print '\n'
        if np.sum(patch != 0):
            if return_patches:
                patch_list += [[corner[:], patch[:]]]
            else:
                patch_list += [corner[:]]

        for idx, corner_dim in enumerate(corner):

            # Advance corner stride
            if idx == 0:
                corner[idx] += patch_shape[idx]

            # Finish patchification
            if idx == len(corner) - 1 and corner[idx] == input_data.shape[-1]:
                finished = True
                continue

            # Push down a dimension.
            if corner[idx] == input_data.shape[idx+1]:
                corner[idx] = 0
                corner[idx+1] += patch_shape[idx+1]

            # Reset patch at edge.
            elif corner[idx] > input_data.shape[idx+1] - patch_shape[idx]:
                corner[idx] = input_data.shape[idx+1] - patch_shape[idx]

    return patch_list

def grab_patch(input_data, corner_list, patch_shape, mask_value=0):

    """ Given a corner coordinate, a patch_shape, and some input_data, returns a patch or array of patches.
    """

    output_patches = np.zeros(((len(corner_list),input_data.shape[1]) + patch_shape))

    for corner_idx, corner in enumerate(corner_list):
        output_slice = [slice(None)]*2 + [slice(corner_dim, corner_dim+patch_shape[idx], 1) for idx, corner_dim in enumerate(corner[1:])]
        output_patches[corner_idx, ...] = input_data[output_slice]

    return output_patches


def insert_patch(input_data, patch, corner):

    patch_shape = patch.shape[1:]

    # print input_data.shape
    # print patch_shape
    # print patch.shape
    # print corner
    patch_slice = [slice(None)]*2 + [slice(corner_dim, corner_dim+patch_shape[idx], 1) for idx, corner_dim in enumerate(corner[1:])]
    # print patch_slice
    # print '\n'

    input_data[patch_slice] = patch

    return

def threshold_binarize(input_data, threshold):

    return (input_data > threshold).astype(float)

def calculate_prediction_msq(label_volume_1, label_volume_2):

    """ Calculate mean-squared error for the predictions folder.
    """

    return

def calculate_prediction_dice(label_volume_1, label_volume_2):

    im1 = np.asarray(label_volume_1).astype(np.bool)
    im2 = np.asarray(label_volume_2).astype(np.bool)

    if im1.shape != im2.shape:
        raise ValueError("Shape mismatch: im1 and im2 must have the same shape.")

    im_sum = im1.sum() + im2.sum()
    if im_sum == 0:
        return empty_score

    # Compute Dice coefficient
    intersection = np.logical_and(im1, im2)

    return 2. * intersection.sum() / im_sum

if __name__ == '__main__':
    pass