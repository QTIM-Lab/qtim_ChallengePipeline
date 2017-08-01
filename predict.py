
import os

import nibabel as nib
import numpy as np
import tables

from model import load_old_model
from image_utils import save_numpy_2_nifti, nifti_2_numpy
from file_util import replace_suffix, nifti_splitext

import multiprocessing
from functools import partial

from joblib import Parallel, delayed

def model_predict_patches_hdf5(data_file, input_data_label, patch_shape, repetitions=16, test_batch_size=200, output_data_label=None, output_shape=None, model=None, model_file=None, output_directory=None, output_name=None, replace_existing=True, merge_labels=True):

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

    if output_data_label is not None:
        truth_list = getattr(data_file.root, output_data_label)

    for case_idx in xrange(total_case_num):

        print 'Working on image.. ', case_idx, 'in', total_case_num

        # Filename for output predictions.
        if output_name == None:
            case_output_name = 'TESTCASE_' + str(case_idx).zfill(3) + '_PREDICT'
        else:
            case_output_name = output_name

        # Destination for predictions. If not in new folder, predict in the same folder as the original image.
        if output_directory is not None:
            output_filepath = os.path.join(output_directory, case_output_name + '.nii.gz')
        else:
            case_directory = casename_list[case_idx]
            # case_directory = np.array_str(np.squeeze(getattr(data_file.root, '_'.join([input_data_label + '_casenames']))[case_idx]))
            output_filepath = os.path.join(case_directory, case_output_name + '.nii.gz')
            print os.path.basename(case_directory)

        # If prediction already exists, skip it. Useful if process is interrupted.
        if os.path.exists(output_filepath) and not replace_existing:
            continue

        # Get data from hdf5
        case_input_data = np.asarray([data_list[case_idx]])
        case_affine = affine_list[case_idx]

        # Get groundtruth if provided.
        if output_data_label is not None:
            case_output = np.asarray([truth_list[case_idx]])

        # Get the shape of the output either from input data, groundtruth, or pre-specification.
        if output_data_label is None and output_shape is None:
            output_shape = list(final_image.shape)
            output_shape[1] = 1
            output_shape = tuple(output_shape)
        elif output_data_label is None:
            pass
        else:
            output_shape = case_output.shape

        predict_patches_one_image(case_input_data)

    data_file.close()

def predict_patches_one_image(input_data, patch_shape, model, output_shape=None, repetitions=1, binarize_probability=.5, ground_truth=None, stack_outputs=False):

    """ Presumes data is in the format (batch_size, channels, dims)
    """

    repetition_offsets = np.linspace(0, 15, repetitions, dtype=int)
    for rep_idx in xrange(repetitions):

        print 'PREDICTION PATCH GRID REPETITION # ..', rep_idx

        offset_slice = [slice(None)]*2 + [slice(repetition_offsets[rep_idx], None, 1)] * (input_data.ndim - 2)
        repatched_image = np.zeros_like(output_data[offset_slice])
        corners_list = patchify_image(input_data[offset_slice], [input_data[offset_slice].shape[1]] + list(patch_shape))

        for corner_list_idx in xrange(0, len(corners_list), test_batch_size):

            corner_batch = corners_list[corner_list_idx:corner_list_idx+test_batch_size]
            input_patches = grab_patch(input_data[offset_slice], corners_list[corner_list_idx:corner_list_idx+test_batch_size], patch_shape)
            prediction = model.predict(input_patches)

            for corner_idx, corner in enumerate(corner_batch):
                insert_patch(repatched_image, prediction[corner_idx, ...], corner)

        if rep_idx == 0:
            output_data = np.copy(repatched_image)
        else:
            # Running Average
            output_data[offset_slice] = output_data[offset_slice] + (1.0 / (rep_idx)) * (repatched_image - output_data[offset_slice])
    
    output_data = np.squeeze(output_data)

    # If output modalities is one, just save the output.
    if output_shape[1] == 1:
        binarized_output_data = threshold_binarize(threshold=binarize_probability, input_data=output_data)
        print 'SUM OF ALL PREDICTION VOXELS', np.sum(binarized_output_data)
        save_numpy_2_nifti(output_data, reference_affine=case_affine, output_filepath=replace_suffix(output_filepath, input_suffix='', output_suffix='-probability'))
        save_numpy_2_nifti(binarized_output_data, reference_affine=case_affine, output_filepath=replace_suffix(output_filepath, input_suffix='', output_suffix='-label'))
        if ground_truth is not None:
            print 'DICE COEFFICIENT', calculate_prediction_dice(binarized_output_data, np.squeeze(ground_truth))
    
    # If multiple output modalities, either stack one on top of the other (e.g. output 3 over output 2 over output 1).
    # or output multiple volumes.
    else:
        if stack_outputs:
            merge_image = threshold_binarize(threshold=binarize_probability, input_data=output_data[0,...])
            print 'SUM OF ALL PREDICTION VOXELS, MODALITY 0', np.sum(merge_image)
            for modality_idx in xrange(1, output_shape[1]):
                print 'SUM OF ALL PREDICTION VOXELS, MODALITY',str(modality_idx), np.sum(output_data[modality_idx,...])
                merge_image[threshold_binarize(threshold=binarize_probability, input_data=output_data[modality_idx,...]) == 1] = modality_idx

            save_numpy_2_nifti(threshold_binarize(threshold=binarize_probability, input_data=output_data[modality,...]), reference_affine=case_affine, output_filepath=output_filepath)
    
        for modality in xrange(output_shape[1]):
            print 'SUM OF ALL PREDICTION VOXELS, MODALITY',str(modality), np.sum(output_data[modality,...])
            binarized_output_data = threshold_binarize(threshold=binarize_probability, input_data=output_data[modality,...])
            save_numpy_2_nifti(output_data[modality,...], reference_affine=case_affine, output_filepath=replace_suffix(output_filepath, input_suffix='', output_suffix='_' + str(modality) + '-probability'))
            save_numpy_2_nifti(binarized_output_data, reference_affine=case_affine, output_filepath=replace_suffix(output_filepath, input_suffix='', output_suffix='_' + str(modality) + '-label'))

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
        patch = grab_patch(input_data, [corner], tuple(patch_shape[1:]))
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

    patch_slice = [slice(None)]*2 + [slice(corner_dim, corner_dim+patch_shape[idx], 1) for idx, corner_dim in enumerate(corner[1:])]
    
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