import os
import glob

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = '3'

import scipy.ndimage as ndimage
import numpy as np
import matplotlib.pyplot as plt
import time
import nibabel as nib
import tables

from mpl_toolkits.mplot3d import Axes3D
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from keras import backend as K
from keras.models import load_model

from qtim_tools.qtim_utilities.format_util import convert_input_2_numpy
from qtim_tools.qtim_utilities.nifti_util import save_numpy_2_nifti

def msq(y_true, y_pred):
    return K.sum(K.pow(y_true - y_pred, 2), axis=None)

def msq_loss(y_true, y_pred):
    return msq(y_true, y_pred)

def dice_coef(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)

def dice_coef_loss(y_true, y_pred):
    return (1 - dice_coef(y_true, y_pred))

def load_old_model(model_file):
    print("Loading pre-trained model")

    custom_objects = {'dice_coef_loss': dice_coef_loss, 'dice_coef': dice_coef, 'msq': msq, 'msq_loss': msq_loss}

    return load_model(model_file, custom_objects=custom_objects)

def read_image_files(image_files, return_affine=False):

    image_list = []
    for i, image_file in enumerate(image_files):
        image_list.append(convert_input_2_numpy(image_file))

    # This is a little clunky.
    if return_affine:
        # This assumes all images share an affine matrix.
        return np.stack([image for image in image_list]), nib.load(image_files[0]).affine
    else:
        return np.stack([image for image in image_list])

def tsne_all_layers(model_file, patch_shape, input_data_filepaths=None, input_directory=None, mask_file=None, mask_labels=[1, 2, 3, 4, 5],layer_range = [5,15], specific_layers=[5,6,9,10,13,14], samples=40000, use_means='all', mode='AFFINE'):

    specific_layers = [1,2,4,5,7,8,10,11,14,15,18,19,22,23]
    # specific_layers = reversed(specific_layers)
    # specific_layers = [23]

    if specific_layers is None:
        specific_layers = xrange(layer_range[0], layer_range[1])

    for layer in specific_layers:
        get_tsne_label(input_data_filepaths, input_directory, model_file, patch_shape, mask_file, mask_labels, layer_output=layer, samples=samples, mode=mode, use_means=use_means)

def get_tsne_label(input_data_filepaths, input_directory, model_file, patch_shape, mask_file, mask_labels, batch_size=400, output_directory=None, output_name=None, replace_existing=True, merge_labels=True, layer_output=14, samples=10000, use_means='mean', mode='AFFINE'):

    # Load model.
    if model_file is None:
        print 'Error. Please provide either a model object or a model filepath.'

    model = load_old_model(model_file)

    if input_directory is None:
        # Get data from hdf5
        # case_input_data, case_affine = read_image_files(input_data_filepaths, return_affine=True)
        # case_input_data = np.expand_dims(case_input_data, 0)
        # if not os.path.exists('layer_' + str(layer_output) + '_samples_' + str(samples) + '.hdf5'):
            # output_prediction_data, output_coordinate_data = write_prediction_to_npy(case_input_data, patch_shape, model, model_batch_size=batch_size, layer_output=layer_output, samples=samples)
        hdf5 = visualize(None, None, layer=layer_output, hdf5 = 'layer_' + str(layer_output) + '_samples_' + str(samples) + '.hdf5', use_means=use_means)
        if hdf5 is not None:
            hdf5.close()

    else:
        input_patches = os.path.join('anatomy_patches_' + str(samples) + '_' + mode + '.hdf5')
        if not os.path.exists(input_patches) or True:
            gather_patches_from_patients(input_directory, input_patches, mask_file, mask_labels, samples, patch_shape, mode)
        if not os.path.exists('layer_' + str(layer_output) + '_' + mode + '_anatomy.hdf5') or True:
            write_prediction_to_npy_anatomy(input_patches, patch_shape, model, layer_output, samples=samples, model_batch_size=batch_size, mode=mode)
        # visualize_anatomy('layer_' + str(layer_output) + '_' + mode + '_anatomy.hdf5', use_means=use_means, layer=layer_output, mode=mode, mask_file=mask_file)
        # tsne_gradient_image(hdf5='layer_' + str(layer_output) + '_' + mode + '_anatomy.hdf5', use_means=use_means, layer=layer_output, mode=mode, mask_file=mask_file)

    # # visualize(output_prediction_data, output_coordinate_data, layer=layer_output,)

def gather_patches_from_patients(input_directory, output_filename, mask_file, mask_labels, samples, patch_shape, mode='AFFINE'):

    # Investigate hdf5 files.
    hdf5_file = tables.open_file(output_filename, mode='w')
    data_shape = (0,4) + patch_shape
    filters = tables.Filters(complevel=5, complib='blosc')
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=samples)
    label_storage = hdf5_file.create_earray(hdf5_file.root, 'labels', tables.Float32Atom(), shape=(0,1), filters=filters, expectedrows=samples)
    coordinate_storage = hdf5_file.create_earray(hdf5_file.root, 'coordinates', tables.Float32Atom(), shape=(0,3), filters=filters, expectedrows=samples)
    casename_storage = hdf5_file.create_earray(hdf5_file.root, 'casenames', tables.StringAtom(256), shape=(0,1), filters=filters, expectedrows=samples)

    patients = glob.glob(os.path.join(input_directory, '*/'))
    print patients
    print os.path.join(input_directory, '*/')

    mask_array = np.expand_dims(np.expand_dims(convert_input_2_numpy(mask_file), 0), 0)
    mask_array = np.repeat(mask_array, 4, 1)

    samples_per_patient = int(samples / len(patients) * 1.5)
    samples_per_mask = samples_per_patient / len(mask_labels)
    print samples_per_mask

    for p in patients:

        print p

        input_data_filepaths = []
        for file in ['FLAIR', 'T2', 'T1', 'T1POST']:
            input_data_filepaths += [os.path.join(p, mode + '_' + file + '.nii')]

        try:
            input_labels = []
            for file in ['TUMOR', 'NECROSIS', 'EDEMA']:
                input_labels += [convert_input_2_numpy(os.path.join(p, mode + '_' + file + '.nii'))]

            case_input_data = read_image_files(input_data_filepaths)
            case_input_data = np.expand_dims(case_input_data, 0)
        except:
            continue

        masked_data = np.copy(case_input_data)

        # Really dumb way to get around invalid patches.
        masked_data[:,:,-patch_shape[0]/2:,...] = 0
        masked_data[:,:,:,-patch_shape[1]/2:,...] = 0
        masked_data[:,:,:,:,-patch_shape[2]/2:] = 0
        masked_data[:,:,:patch_shape[0]/2,...] = 0
        masked_data[:,:,:,:patch_shape[1]/2,...] = 0
        masked_data[:,:,:,:,:patch_shape[2]/2] = 0

        non_zero_corners = np.nonzero(masked_data)
        non_zero_corners = np.array(non_zero_corners)[1:,:].T
        np.random.shuffle(non_zero_corners)

        for corner_idx in xrange(samples_per_patient):

            corner = non_zero_corners[corner_idx]
            output_slice = [slice(None)]*2 + [slice(corner_dim-patch_shape[idx]/2, corner_dim+patch_shape[idx]/2, 1) for idx, corner_dim in enumerate(corner[1:])]
            data_storage.append(case_input_data[output_slice])
            coordinate_storage.append(corner[1:].reshape(1,3))
            casename_storage.append(np.array(p)[np.newaxis][np.newaxis])

            label_value = None
            if input_labels[0][int(corner[1]), int(corner[2]), int(corner[3])] == 1:
                label_value = np.array([6]).reshape(1,1)
            if input_labels[1][int(corner[1]), int(corner[2]), int(corner[3])] == 1:
                label_value = np.array([7]).reshape(1,1)
            if input_labels[2][int(corner[1]), int(corner[2]), int(corner[3])] == 1:
                label_value = np.array([8]).reshape(1,1)

            if label_value is None:
                label_value = mask_array[0,0,int(corner[1]), int(corner[2]), int(corner[3])].reshape(1,1)

            print label_value

            label_storage.append(label_value)

        # for m in mask_labels:

        #     label_data = np.copy(masked_data)

        #     try:
        #         label_data[mask_array != m] = 0
                
        #         # Get all possible centers.
        #         non_zero_corners = np.nonzero(label_data)
        #         non_zero_corners = np.array(non_zero_corners)[1:,:].T
        #         np.random.shuffle(non_zero_corners)

        #         for corner_idx in xrange(samples_per_mask):

        #             corner = non_zero_corners[corner_idx]
        #             output_slice = [slice(None)]*2 + [slice(corner_dim-patch_shape[idx]/2, corner_dim+patch_shape[idx]/2, 1) for idx, corner_dim in enumerate(corner[1:])]
        #             data_storage.append(case_input_data[output_slice])
        #             label_storage.append(np.array(m).reshape(1,1))
        #             coordinate_storage.append(corner[1:].reshape(1,3))
        #             casename_storage.append(np.array(p)[np.newaxis][np.newaxis])
        #     except:
        #         print 'ERROR at...', m
                
    hdf5_file.close()

def create_hdf5_file(hdf5_filepath, shape, samples):

    # Investigate hdf5 files.
    hdf5_file = tables.open_file(hdf5_filepath, mode='w')

    data_shape = (0,) + shape[1:]

    # Investigate this line.
    # Compression levels = complevel. No compression = 0
    # Compression library = Method of compresion.
    filters = tables.Filters(complevel=5, complib='blosc')

    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=samples)

    coordinate_storage = hdf5_file.create_earray(hdf5_file.root, 'coordinates', tables.Float32Atom(), shape=(0,3), filters=filters, expectedrows=samples)

    return hdf5_file, data_storage, coordinate_storage

def write_prediction_to_npy(input_data, patch_shape, model, model_batch_size=200, layer_output=14, samples=15000, save=False, mode='AFFINE'):

    get_specific_layer_output = K.function([model.layers[0].input], [model.layers[layer_output].output])
    output_shape = model.layers[layer_output].output_shape
    output_hdf5, data_storage, coordinate_storage = create_hdf5_file('layer_' + str(layer_output) + '_' + mode + '_samples_' + str(samples) + '.hdf5', output_shape, samples)

    # Really dumb way to get around invalid patches.
    input_data[:,:,-patch_shape[0]/2:,...] = 0
    input_data[:,:,:,-patch_shape[1]/2:,...] = 0
    input_data[:,:,:,:,-patch_shape[2]/2:] = 0
    input_data[:,:,:patch_shape[0]/2,...] = 0
    input_data[:,:,:,:patch_shape[1]/2,...] = 0
    input_data[:,:,:,:,:patch_shape[2]/2] = 0

    # Get all possible centers.
    non_zero_corners = np.nonzero(input_data)
    non_zero_corners = np.array(non_zero_corners)[1:,:].T
    np.random.shuffle(non_zero_corners)

    for corner_idx in xrange(0, non_zero_corners.shape[0], model_batch_size):

        if corner_idx > samples:
            break

        corners_list = non_zero_corners[corner_idx:corner_idx+model_batch_size]

        input_patches = grab_patch(input_data, corners_list, patch_shape)

        prediction = get_specific_layer_output([input_patches])[0]
        midpoints = np.array([[corner[1], corner[2], corner[3]]  for corner in corners_list])

        data_storage.append(prediction)
        coordinate_storage.append(midpoints)

        print corner_idx

    output_hdf5.close()
    return None, None

def write_prediction_to_npy_anatomy(input_hdf5_file, patch_shape, model, layer_output, model_batch_size=200, samples=15000, mode='AFFINE'):

    output_shape = model.layers[layer_output].output_shape
    hdf5_file = tables.open_file('layer_' + str(layer_output) + '_' + mode + '_anatomy.hdf5', mode='w')
    data_shape = (0,) + output_shape[1:]
    filters = tables.Filters(complevel=5, complib='blosc')
    data_storage = hdf5_file.create_earray(hdf5_file.root, 'data', tables.Float32Atom(), shape=data_shape, filters=filters, expectedrows=samples)
    label_storage = hdf5_file.create_earray(hdf5_file.root, 'labels', tables.Float32Atom(), shape=(0,1), filters=filters, expectedrows=samples)
    coordinate_storage = hdf5_file.create_earray(hdf5_file.root, 'coordinates', tables.Float32Atom(), shape=(0,3), filters=filters, expectedrows=samples)
    casename_storage = hdf5_file.create_earray(hdf5_file.root, 'casenames', tables.StringAtom(256), shape=(0,1), filters=filters, expectedrows=samples)


    get_specific_layer_output = K.function([model.layers[0].input], [model.layers[layer_output].output])
    input_hdf5 = tables.open_file(input_hdf5_file, "r")
    data = getattr(input_hdf5.root, 'data')
    labels = getattr(input_hdf5.root, 'labels')
    coordinates = getattr(input_hdf5.root, 'coordinates')
    casenames = getattr(input_hdf5.root, 'casenames')

    for idx in xrange(0, data.shape[0], model_batch_size):

        print idx
        prediction = get_specific_layer_output([data[idx:idx+model_batch_size]])[0]
        data_storage.append(prediction)
        label_storage.append(labels[idx:idx+model_batch_size])
        coordinate_storage.append(coordinates[idx:idx+model_batch_size])
        casename_storage.append(casenames[idx:idx+model_batch_size])

    hdf5_file.close()
    input_hdf5.close()

    return 'layer_' + str(layer_output) + '_anatomy.hdf5'

def grab_patch(input_data, corner_list, patch_shape, mask_value=0):

    """ Given a corner coordinate, a patch_shape, and some input_data, returns a patch or array of patches.
    """

    output_patches = np.zeros(((len(corner_list),input_data.shape[1]) + patch_shape))

    for corner_idx, corner in enumerate(corner_list):
        output_slice = [slice(None)]*2 + [slice(corner_dim-patch_shape[idx]/2, corner_dim+patch_shape[idx]/2, 1) for idx, corner_dim in enumerate(corner[1:])]
        output_patches[corner_idx, ...] = input_data[output_slice]

    return output_patches

def visualize_anatomy(hdf5, use_means='mean', calc_tsne=False, calc_pca=False, save_plot=True, show_plot=False, layer=14, perplexity=50, mode='AFFINE', mask_file=''):


    print 'starting anatomy viz....'
    np.set_printoptions(suppress=True)

    input_hdf5 = tables.open_file(hdf5, "r")
    labels = getattr(input_hdf5.root, 'labels')
    coords = getattr(input_hdf5.root, 'coordinates')

    if calc_tsne or not os.path.exists('tsne_anatomy' + str(layer) + '_' + mode + '_perplexity_' + str(perplexity) + '_' + use_means + '.npy'):

        data = getattr(input_hdf5.root, 'data')
        # input_hdf5.close()   

        if use_means == 'mean':
            data = np.reshape(data, (data.shape[0], data.shape[1], -1))
            data = np.mean(data, axis=2)
        else:
            data = np.reshape(data, (data.shape[0], -1))

        print data.shape

        if data.shape[1] < 100:
            pass
        elif calc_pca or not os.path.exists('pca_' + str(layer) + '_' + mode + '_' + use_means + '.npy'):
            pca = PCA(n_components=100)
            pca_result = pca.fit_transform(data)
            print pca_result.shape

            print 'Explained variation per principal component: {}'.format(pca.explained_variance_ratio_)

            data = pca_result

            np.save('pca_anatomy' + str(layer) + '_' + mode + '_' + use_means + '.npy', data)
        else:
            data = np.load('pca_anatomy' + str(layer) + '_' + mode + '_' + use_means + '.npy')

        time_start = time.time()

        print 'Starting tsne'

        import bhtsne

        tsne_results = bhtsne.run_bh_tsne(data, no_dims=2, initial_dims=data.shape[1], verbose=True, perplexity=50, theta=0.5, randseed=-1, max_iter=1000)

        np.save('tsne_anatomy' + str(layer) + '_' + mode + '_perplexity_' + str(perplexity) + '_' + use_means + '.npy', tsne_results)

        print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)
    else:
        tsne_results = np.load('tsne_anatomy' + str(layer) + '_' + mode + '_perplexity_' + str(perplexity) + '_' + use_means + '.npy')

    print 'Saving plot..'

    if save_plot or show_plot or not os.path.exists('tsne_plot_anatomy_' + str(layer) + '_' + mode + '_perplexity_' + str(perplexity) + '_' + use_means + '.png'):

        # 2D
        fig = plt.figure(figsize=(16.0, 10.0))
        sc = plt.scatter(tsne_results[:,0], tsne_results[:,1])
        # plt.colorbar(sc)
        if show_plot:
            plt.show()
        plt.savefig('tsne_plot_anatomy_' + str(layer) + '_' + mode + '_perplexity_' + str(perplexity) + '_no_color' + use_means + '.png', bbox_inches='tight')
        plt.clf()

        # 3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(tsne_results[:,0], tsne_results[:,1], tsne_results[:,2], c=k_clusters)
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()


    k_clusters = AgglomerativeClustering(n_clusters=9).fit_predict(tsne_results)
    fig = plt.figure(figsize=(16.0, 10.0))
    plt.scatter(tsne_results[:,0], tsne_results[:,1], c=k_clusters)
    # plt.show()

    image_array = convert_input_2_numpy(mask_file)
    output_array = np.zeros_like(image_array)

    for idx, coordinate in enumerate(coords):
        output_array[int(coordinate[0]), int(coordinate[1]), int(coordinate[2])] = k_clusters[idx] + 1

    B = ndimage.maximum_filter(output_array, 3)
    B[output_array != 0] = output_array[output_array != 0]

    save_numpy_2_nifti(B, mask_file, 'tsne_' + str(layer) + '_' + mode + '_perplexity_' + str(perplexity) + '_' + use_means + '-label.nii.gz')

def tsne_gradient_image(layer, hdf5=None, mode='AFFINE', perplexity=50, use_means='mean', mask_file=None):

    input_hdf5 = tables.open_file(hdf5, "r")
    coords = getattr(input_hdf5.root, 'coordinates')

    try:
        tsne_results = np.load('tsne_anatomy' + str(layer) + '_' + mode + '_perplexity_' + str(perplexity) + '_' + use_means + '.npy')

        image_array = convert_input_2_numpy(mask_file)
        x_array = np.zeros_like(image_array)
        y_array = np.zeros_like(image_array)

        for idx, coordinate in enumerate(coords):
            x_array[int(coordinate[0]), int(coordinate[1]), int(coordinate[2])] = tsne_results[idx, 0]
            y_array[int(coordinate[0]), int(coordinate[1]), int(coordinate[2])] = tsne_results[idx, 1]

        x_array = x_array - np.min(x_array)
        y_array = y_array - np.min(y_array)
        B = ndimage.maximum_filter(x_array, 5)
        B[x_array != 0] = x_array[x_array != 0]
        x_array = B
        B = ndimage.maximum_filter(y_array, 5)
        B[y_array != 0] = y_array[y_array != 0]
        y_array = B

        save_numpy_2_nifti(x_array, mask_file, 'tsne_' + str(layer) + '_' + mode + '_perplexity_' + str(perplexity) + '_' + use_means + 'X.nii.gz')
        save_numpy_2_nifti(y_array, mask_file, 'tsne_' + str(layer) + '_' + mode + '_perplexity_' + str(perplexity) + '_' + use_means + 'Y.nii.gz')
    except:
        print 'Gradient image not computed'

    input_hdf5.close()

    return

def visualize(input_data=None, input_coordinates=None, load=False, use_means='mean', calc_tsne=False, calc_pca=False, save_plot=True, show_plot=False, layer=14, hdf5=None, perplexity=50):

    print 'starting viz...'
    np.set_printoptions(suppress=True)

    if hdf5 is not None:
        input_hdf5 = tables.open_file(hdf5, "r")
        coords = getattr(input_hdf5.root, 'coordinates')
    else:
        input_hdf5 = None
        coords = getattr(input_coordinates.root, 'coordinates')

    if calc_tsne or not os.path.exists('tsne_' + str(layer) + '_perplexity_' + str(perplexity) + '_' + use_means + '.npy'):

        data = getattr(input_hdf5.root, 'data')

        if use_means == 'mean':
            data = np.reshape(data, (data.shape[0], data.shape[1], -1))
            data = np.mean(data, axis=2)
        else:
            data = np.reshape(data, (data.shape[0], -1))

        print data.shape

        if data.shape[1] < 100:
            pass
        elif calc_pca or not os.path.exists('pca_' + str(layer) + '_' + use_means + '.npy'):
            pca = PCA(n_components=100)
            pca_result = pca.fit_transform(data)
            print pca_result.shape

            print 'Explained variation per principal component: {}'.format(pca.explained_variance_ratio_)

            data = pca_result

            np.save('pca_' + str(layer) + '_' + use_means + '.npy', data)
        else:
            data = np.load('pca_' + str(layer) + '_' + use_means + '.npy')

        time_start = time.time()

        print 'Starting tsne'

        import bhtsne

        tsne_results = bhtsne.run_bh_tsne(data, no_dims=2, initial_dims=data.shape[1], verbose=True, perplexity=50, theta=0.5, randseed=-1, max_iter=1000)

        np.save('tsne_' + str(layer) + '_perplexity_' + str(perplexity) + '_' + use_means + '.npy', tsne_results)

        print 't-SNE done! Time elapsed: {} seconds'.format(time.time()-time_start)
    else:
        tsne_results = np.load('tsne_' + str(layer) + '_perplexity_' + str(perplexity) + '_' + use_means + '.npy')

    print 'Saving plot..'

    if save_plot or show_plot or not os.path.exists('tsne_plot_' + str(layer) + '_perplexity_' + str(perplexity) + '_' + use_means + '.png'):

        k_clusters = AgglomerativeClustering(n_clusters=10).fit_predict(tsne_results)

        # 2D
        fig = plt.figure(figsize=(16.0, 10.0))
        plt.scatter(tsne_results[:,0], tsne_results[:,1], c=k_clusters)
        if show_plot:
            plt.show()
        plt.savefig('tsne_plot_' + str(layer) + '_perplexity_' + str(perplexity) + '_' + use_means + '.png', bbox_inches='tight')
        plt.clf()

        # 3D
        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        # ax.scatter(tsne_results[:,0], tsne_results[:,1], tsne_results[:,2], c=k_clusters)
        # ax.set_xlabel('X Label')
        # ax.set_ylabel('Y Label')
        # ax.set_zlabel('Z Label')
        # plt.show()

    file ='/mnt/jk489/sharedfolder/BRATS2017/Test_Case_2/Brats17_2013_24_1/FLAIR_pp.nii.gz'
    image_array = convert_input_2_numpy(file)
    output_array = np.zeros_like(image_array)

    print output_array.shape

    for idx, coordinate in enumerate(coords):
        output_array[int(coordinate[0]), int(coordinate[1]), int(coordinate[2])] = k_clusters[idx] + 1

    B = ndimage.maximum_filter(output_array, 3)
    B[output_array != 0] = output_array[output_array != 0]

    save_numpy_2_nifti(B, file, 'tsne_' + str(layer) + '_perplexity_' + str(perplexity) + '_' + use_means + '-label.nii.gz')

    return input_hdf5

if __name__ == '__main__':

    input_files = ['FLAIR_pp.nii.gz', 'T2_pp.nii.gz', 'T1_pp.nii.gz', 'T1post_pp.nii.gz']
    input_files = [os.path.join('/mnt/jk489/sharedfolder/BRATS2017/Test_Case_2/Brats17_2013_24_1', file) for file in input_files]

    model = 'brats_edema_pregenerated_.h5'

    input_directory = '/mnt/jk489/sharedfolder/BRATS2017/Atlas_Registrations/5_labels'
    mask_file = '/mnt/jk489/sharedfolder/BRATS2017/Atlas_Registrations/5_labels/brain_atlas-label.nii.gz'

    for mode in ['AFFINE', 'DEFORMABLE']:
        tsne_all_layers(model, (16, 16, 16), input_directory=input_directory, mask_file=mask_file, mode=mode, use_means='mean')