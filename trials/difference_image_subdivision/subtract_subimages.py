import os
from astropy.io import fits
from scipy.signal import convolve2d
from pyDANDIA.read_images_stage5 import open_reference, open_images, open_data_image
from multiprocessing import Pool
from scipy.ndimage.interpolation import shift
import multiprocessing as mp
import numpy as np

def subtract_images(data_image, reference_image, kernel, kernel_size, bkg_kernel, mask = None):
    model_image = convolve2d(reference_image, kernel, mode='same')
    model_image[data_image==0]=-bkg_kernel
    difference_image = model_image - data_image + bkg_kernel
    difference_image = difference_image
    return difference_image, model_image

def subtract_images_renorm(data_image, reference_image, kernel, kernel_size, bkg_kernel, mask = None, ref_subset = None):
    #add zero margin
    # extend image size for convolution and kernel solution
    model_image = convolve2d(reference_image, kernel, mode='same')
    if ref_subset != None:
        model_image = model_image[ref_subset[0]:ref_subset[1],ref_subset[2]:ref_subset[3]]
    model_image[data_image==0] = -bkg_kernel
    difference_image = model_image - data_image + bkg_kernel
    difference_image = difference_image - np.median(difference_image)
    difference_image = difference_image/np.sum(kernel)
    return difference_image

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

def nearest_kernel_index(reduction_metadata, center_x, center_y):
    tile_center_x = []
    tile_center_y = []
    for substamp_idx in range(len(reduction_metadata.stamps[1])):
        tile_center_x.append(((-float(reduction_metadata.stamps[1][substamp_idx]['X_MIN'])+float(reduction_metadata.stamps[1][substamp_idx]['X_MAX']))*0.5+float(reduction_metadata.stamps[1][substamp_idx]['X_MIN'])))
        tile_center_y.append(((-float(reduction_metadata.stamps[1][substamp_idx]['Y_MIN'])+float(reduction_metadata.stamps[1][substamp_idx]['Y_MAX']))*0.5+float(reduction_metadata.stamps[1][substamp_idx]['Y_MIN'])))
    r = (np.array(tile_center_x)-center_x)**2+(np.array(tile_center_y)-center_y)**2
    return np.argsort(r)

def subtraction_stamp_grid(reduction_metadata, reference_shape):
    #generate subtraction stamps 1/3 of the original subimage
    x_size = float(reduction_metadata.stamps[1][0]['X_MAX'])
    y_size = float(reduction_metadata.stamps[1][0]['Y_MAX'])
    sub_x_size = int(x_size//3)
    sub_y_size = int(y_size//3)
    sub_center_x = []
    sub_center_y = []
    for xval in range(int(x_size//6),reference_shape[1],sub_x_size):
        for yval in range(int(y_size//6),reference_shape[0],sub_y_size):
            sub_center_x.append(xval)
            sub_center_y.append(yval)
    return sub_center_x,sub_center_y, sub_x_size, sub_y_size    

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
    reference_stamps = open_reference_subimages(setup, reduction_metadata, kernel_size, max_adu, maxshift)
    kernel_stamps = np.load(os.path.join(kernel_directory_path,'kernel_multi_'+image_name+'.npy'))
    scalef = []
    for kernel_idx in range(len(kernel_stamps)):
        kernel_matrix, bkg_kernel, kernel_uncertainty = kernel_stamps[kernel_idx]
        scalef.append(np.sum(kernel_matrix))
    med_all=np.median(scalef)
    data_image_directory = reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]
    x_max = int(reduction_metadata.stamps[1][-1]['X_MAX'])
    y_max = int(reduction_metadata.stamps[1][-1]['Y_MAX'])
    data_image_directory = reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]
    assembled =  np.zeros((y_max, x_max))
    reference_image_name = str(reduction_metadata.data_architecture[1]['REF_IMAGE'][0])
    reference_image_directory = str(reduction_metadata.data_architecture[1]['REF_PATH'][0])
    data_image_directory = reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]
    reference_image = fits.open(os.path.join(reference_image_directory, reference_image_name))
    reference_data_array = reference_image[0].data
    row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == image_name)[0][0] 
    data_image = fits.open(os.path.join(data_image_directory,image_name))
    unshifted_data_image_array = np.copy(data_image[0].data)
    xshift, yshift = -reduction_metadata.images_stats[1][row_index]['SHIFT_X'],-reduction_metadata.images_stats[1][row_index]['SHIFT_Y'] 
    data_image[0].data = shift(data_image[0].data, (-yshift,-xshift), cval=0.)
    data_image_array = data_image[0].data 
    #open all reference images
    umatrix_stamps, kernel_size, max_adu, maxshift = np.load(os.path.join(kernel_directory_path,'unweighted_u_matrix_subimages.npy'))
    kernel_stamps = np.load(os.path.join(kernel_directory_path,'kernel_multi_'+image_name+'.npy'))
    data_image_directory = reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]
    x_max = int(reduction_metadata.stamps[1][-1]['X_MAX'])
    y_max = int(reduction_metadata.stamps[1][-1]['Y_MAX'])
    sub_center_x, sub_center_y, x_size, y_size = subtraction_stamp_grid(reduction_metadata, (x_max, y_max))
    #define kernel center for interpolation/nearest neighbour
    #test case: single subimage, e.g. at 500,500 with 40x40:
    for idx in range(len(sub_center_x)):
        substamp_indices = nearest_kernel_index(reduction_metadata, sub_center_x[idx], sub_center_y[idx])
        #contains the sorted list of nearest kernels -> test which is best...
        subset2 = [sub_center_y[idx] - int(y_size//2), sub_center_y[idx] + int(y_size//2)+1 , sub_center_x[idx] - int(x_size//2), sub_center_x[idx] + int(x_size//2)+1 ]        
     
 
        #decide how to extend reference
        subset2_reference = []
        subset2_revert = []
        if sub_center_y[idx] - int(y_size // 2) - kernel_size >0: 
            subset2_reference.append(sub_center_y[idx] - int(y_size // 2) - kernel_size)
            subset2_revert.append(kernel_size)
        else:
            subset2_reference.append(sub_center_y[idx] - y_size//2)
            subset2_revert.append(0)

        if sub_center_y[idx] + int(y_size // 2) + kernel_size < y_max:
            subset2_reference.append(sub_center_y[idx] + int(y_size // 2) + kernel_size +1)
            subset2_revert.append(-kernel_size)
        else:
            subset2_reference.append(sub_center_y[idx] + int(y_size // 2))
            subset2_revert.append(None)	

        if sub_center_x[idx] - int(x_size // 2) - kernel_size >0: 
            subset2_reference.append(sub_center_x[idx] - int(x_size // 2) - kernel_size)
            subset2_revert.append(kernel_size)
        else:
            subset2_reference.append(sub_center_x[idx] - x_size//2)
            subset2_revert.append(0)

        if sub_center_x[idx] + int(x_size // 2) + kernel_size < x_max:
            subset2_reference.append(sub_center_x[idx] + int(x_size // 2) + kernel_size +1)
            subset2_revert.append(-kernel_size)
        else:
            subset2_reference.append(sub_center_x[idx] + int(x_size // 2) )
            subset2_revert.append(None)
        
        scalef = []
        for kernel_idx in range(min(4,int(len(sub_center_x)))):
            substamp_idx = substamp_indices[kernel_idx]
            kernel_matrix, bkg_kernel, kernel_uncertainty = kernel_stamps[substamp_idx]
            substamp_idx = substamp_indices[kernel_idx]
            scalef.append(np.sum(kernel_matrix))
        substamp_idx = substamp_indices[np.argmin((np.array(scalef)-med_all)**2)]
        subset = [int(reduction_metadata.stamps[1][substamp_idx]['Y_MIN']), int(reduction_metadata.stamps[1][substamp_idx]['Y_MAX']), int(reduction_metadata.stamps[1][substamp_idx]['X_MIN']),int(reduction_metadata.stamps[1][substamp_idx]['X_MAX'])] 
        bkg=np.median( reference_data_array[subset2_reference[0]:subset2_reference[1],subset2_reference[2]:subset2_reference[3]])
        reference_subimage = reference_data_array[subset2_reference[0]:subset2_reference[1],subset2_reference[2]:subset2_reference[3]]-bkg
        #select corresponding image
        kernel_matrix, bkg_kernel, kernel_uncertainty = kernel_stamps[substamp_idx]
        difference_image = subtract_images_renorm(data_image_array[subset2[0]:subset2[1],subset2[2]:subset2[3]] - np.median(unshifted_data_image_array[subset2[0]:subset2[1],subset2[2]:subset2[3]]), reference_subimage, kernel_matrix, kernel_size, bkg_kernel, ref_subset = subset2_revert)

        assembled[subset2[0]:subset2[1],subset2[2]:subset2[3]] = difference_image

    new_header = fits.Header()
    difference_image_hdu = fits.PrimaryHDU(assembled ,header=new_header)
    difference_image_hdu.writeto(os.path.join(diffim_directory_path,'master_diff_'+image_name),overwrite = True)
     

