import os
import pickle
from random import shuffle

import numpy as np

from data_utils import pickle_dump, pickle_load

def get_data_generator(data_file, batch_size, data_labels, augmentations=None):

    if isinstance(data_labels, basestring):
        data_labels = [data_labels]

    num_steps = getattr(data_file.root, data_labels[0]).shape[0]
    output_data_generator = data_generator(data_file, range(num_steps), data_labels=data_labels, batch_size=batch_size, augmentations=augmentations)

    return output_data_generator, num_steps // batch_size

def data_generator(data_file, index_list, data_labels, batch_size=1, augmentations=None):

    """ TODO: Investigate how generators even work?! And yield.
    """

    while True:
        data_lists = [[] for i in data_labels]
        shuffle(index_list)

        for index in index_list:

            add_data(data_lists, data_file, index, data_labels, augmentations)

            if len(data_lists[0]) == batch_size:

                yield tuple([np.asarray(data_list) for data_list in data_lists])
                data_lists = [[] for i in data_labels]

def add_data(data_lists, data_file, index, data_labels, augmentations=None):

    for data_idx, data_label in enumerate(data_labels):
        data = getattr(data_file.root, data_label)[index]
        data_lists[data_idx].append(data)

def get_patch_data_generator(data_file, batch_size, data_labels, patch_shape, patch_multiplier=1, roi_ratio=.7):

    if isinstance(data_labels, basestring):
        data_labels = [data_labels]

    num_steps = getattr(data_file.root, data_labels[0]).shape[0]
    output_data_generator = patch_data_generator(data_file, range(num_steps), data_labels=data_labels, batch_size=batch_size, patch_shape=patch_shape, patch_multiplier=patch_multiplier, roi_ratio=roi_ratio)

    return output_data_generator, (num_steps * patch_multiplier) // batch_size

def patch_data_generator(data_file, index_list, data_labels, batch_size=1, patch_shape=(16,16,16), patch_multiplier=5, roi_ratio=.7):

    """ TODO: Investigate how generators even work?! And yield.
    """

    while True:
        data_lists = [[] for i in data_labels]
        shuffle(index_list)

        for index in index_list:

            add_patch_data(data_lists, data_file, index, data_labels, patch_shape, roi_ratio, patch_multiplier=patch_multiplier)

            if len(data_lists[0]) == batch_size:

                yield tuple([np.asarray(data_list) for data_list in data_lists])
                data_lists = [[] for i in data_labels]

def add_patch_data(data_lists, data_file, index, data_labels, patch_shape, roi_ratio, patch_multiplier):

    input_data = getattr(data_file.root, 'input_modalities')[index]
    ground_truth = getattr(data_file.root, 'ground_truth')[index]

    brainmask = np.load(np.array_str(np.squeeze(getattr(data_file.root, 'input_modalities_brainmask')[index])))
    roimask = np.load(np.array_str(np.squeeze(getattr(data_file.root, 'input_modalities_roimask')[index])))

    patch_input = np.zeros(input_data.shape[0:2] + patch_shape)
    patch_gt = np.zeros(ground_truth.shape[0:2] + patch_shape)

    for patch_idx in xrange(patch_multiplier):
        if np.random.uniform > roi_ratio:
            curr_idx = brainmask[np.random.randint(0, len(brainmask))]
        else:
            curr_idx = roimask[np.random.randint(0, len(roimask))]
                                 
        patch_input = input_data[:,(curr_idx-patch_shape[0]/2)[0]:(curr_idx+patch_shape[0]/2)[0],(curr_idx-patch_shape[1]/2)[1]:(curr_idx+patch_shape[1]/2)[1],(curr_idx-patch_shape[2]/2)[2]:(curr_idx+patch_shape[2]/2)[2]]    
        patch_seg = ground_truth[:,(curr_idx-patch_shape[0]/2)[0]:(curr_idx+patch_shape[0]/2)[0],(curr_idx-patch_shape[1]/2)[1]:(curr_idx+patch_shape[1]/2)[1],(curr_idx-patch_shape[2]/2)[2]:(curr_idx+patch_shape[2]/2)[2]]

        data_lists[0].append(patch_input)
        data_lists[1].append(patch_seg)

def shuffle_all_indices(data_file):

    num_items = getattr(data_file.root, data_labels[0]).shape[0]

    for idx in xrange(num_items):
        brainmask = np.load(np.array_str(np.squeeze(getattr(data_file.root, 'input_modalities_brainmask')[idx])))
        roimask = np.load(np.array_str(np.squeeze(getattr(data_file.root, 'input_modalities_roimask')[idx])))
        np.random.seed(0)
        np.shuffle(brainmask)
        np.random.seed(0)
        np.shuffle(roimask)
        np.save(brainmask, np.array_str(np.squeeze(getattr(data_file.root, 'input_modalities_brainmask')[idx])))
        np.save(roimask, np.array_str(np.squeeze(getattr(data_file.root, 'input_modalities_roimask')[idx])))


