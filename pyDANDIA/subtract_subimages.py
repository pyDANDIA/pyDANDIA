import os
from astropy.io import fits
from scipy.signal import convolve2d
from pyDANDIA.read_images_stage5 import open_reference, open_images, open_data_image
from multiprocessing import Pool
import multiprocessing as mp
import numpy as np

def subtract_images(data_image, reference_image, kernel, kernel_size, bkg_kernel, mask = None):
    model_image = convolve2d(reference_image, kernel, mode='same')
    model_image[data_image==0]=-bkg_kernel
    difference_image = model_image - data_image + bkg_kernel
    difference_image = difference_image
    return difference_image, model_image

def open_reference_subimages(setup, reduction_metadata, kernel_size, max_adu, maxshift):
    reference_stamps = []
    reference_image_name = str(reduction_metadata.data_architecture[1]['REF_IMAGE'][0])
    reference_image_directory = str(reduction_metadata.data_architecture[1]['REF_PATH'][0])
    #load all reference subimages
    ref_image1 = fits.open(os.path.join(reference_image_directory, reference_image_name))
    for substamp_idx in range(len(reduction_metadata.stamps[1])):
        subset_slice = [int(reduction_metadata.stamps[1][substamp_idx]['Y_MIN']),int(reduction_metadata.stamps[1][substamp_idx]['Y_MAX']),int(reduction_metadata.stamps[1][substamp_idx]['X_MIN']),int(reduction_metadata.stamps[1][substamp_idx]['X_MAX'])]       
        reference_image, bright_reference_mask, reference_image_unmasked = open_reference(setup, reference_image_directory, reference_image_name, kernel_size, max_adu, ref_extension = 0, log = None, central_crop = maxshift, subset = subset_slice,ref_image1=ref_image1, min_adu = None, subtract = True)
        reference_stamps.append([np.copy(reference_image), bright_reference_mask, np.copy(reference_image_unmasked)])       
    return reference_stamps

def subtract_subimage(setup, kernel_directory_path, image_name,reduction_metadata):
    #open all reference images
    umatrix_stamps, kernel_size, max_adu, maxshift = np.load(os.path.join(kernel_directory_path,'unweighted_u_matrix_subimages.npy'))
    reference_stamps = open_reference_subimages(setup, reduction_metadata, kernel_size, max_adu, maxshift)
    kernel_stamps = np.load(os.path.join(kernel_directory_path,'kernel_multi_'+image_name+'.npy'))
    data_image_directory = reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]
    x_max = int(reduction_metadata.stamps[1][-1]['X_MAX'])
    y_max = int(reduction_metadata.stamps[1][-1]['Y_MAX'])

    assembled =  np.zeros((y_max, x_max))
    for substamp_idx in range(len(reduction_metadata.stamps[1])):
        subset_slice = [int(reduction_metadata.stamps[1][substamp_idx]['Y_MIN']),int(reduction_metadata.stamps[1][substamp_idx]['Y_MAX']),int(reduction_metadata.stamps[1][substamp_idx]['X_MIN']),int(reduction_metadata.stamps[1][substamp_idx]['X_MAX'])]

        kernel_matrix, bkg_kernel, kernel_uncertainty = kernel_stamps[substamp_idx]
        row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == image_name)[0][0]
        x_shift, y_shift = -reduction_metadata.images_stats[1][row_index]['SHIFT_X'],-reduction_metadata.images_stats[1][row_index]['SHIFT_Y'] 
        data_image, data_image_unmasked = open_data_image(setup, data_image_directory, image_name, reference_stamps[substamp_idx][1], kernel_size, max_adu, xshift = x_shift, yshift = y_shift, sigma_smooth = 0, central_crop = maxshift, subset = subset_slice,min_adu = None, subtract = True)
        missing_data_mask = (data_image == 0.)
        difference_image, model_image = subtract_images(data_image_unmasked, reference_stamps[substamp_idx][2], kernel_matrix, kernel_size, bkg_kernel)
        #mask overlap
        nonzero_diffim=difference_image != 0.
        assembled[subset_slice[0]:subset_slice[1], subset_slice[2]:subset_slice[3]][nonzero_diffim] = difference_image[nonzero_diffim]
        diffim_directory_path = os.path.join(setup.red_dir, 'diffim')

    new_header = fits.Header()
    difference_image_hdu = fits.PrimaryHDU(assembled ,header=new_header)
    difference_image_hdu.writeto(os.path.join(diffim_directory_path,'master_diff_'+str(substamp_idx)+image_name),overwrite = True)

def subtract_subimage_interp(setup, kernel_directory_path, image_name,reduction_metadata):
    #open all reference images
    umatrix_stamps, kernel_size, max_adu, maxshift = np.load(os.path.join(kernel_directory_path,'unweighted_u_matrix_subimages.npy'))

    reference_stamps = open_reference_regular_grid(setup, reduction_metadata, kernel_size, max_adu, maxshift)

    kernel_stamps = np.load(os.path.join(kernel_directory_path,'kernel_multi_'+image_name+'.npy'))
    data_image_directory = reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]
    x_max = int(reduction_metadata.stamps[1][-1]['X_MAX'])
    y_max = int(reduction_metadata.stamps[1][-1]['Y_MAX'])

    assembled =  np.zeros((y_max, x_max))
    #define kernel center for interpolation/nearest neighbour
    tile_center = []
    for substamp_idx in range(len(reduction_metadata.stamps[1])):
        tile_center.append([-float(reduction_metadata.stamps[1][substamp_idx]['Y_MIN'])+float(reduction_metadata.stamps[1][substamp_idx]['Y_MAX']),-float(reduction_metadata.stamps[1][substamp_idx]['X_MIN'])+float(reduction_metadata.stamps[1][substamp_idx]['X_MAX'])])

        print subset_slice[-1]

