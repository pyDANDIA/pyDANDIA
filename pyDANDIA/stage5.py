######################################################################
#
# stage5.py - Fifth stage of the pipeline. Calculate kernel solution
# and optionally subtract

#
# dependencies:
#      numpy 1.8+
#      astropy 1.0+
#      scipy 1.0+
######################################################################
import os
import numpy as np
from astropy.io import fits
from scipy.signal import convolve2d
from read_images_stage5 import open_reference,  open_images, open_data_image
from scipy.ndimage.filters import gaussian_filter
import convolution

import sys

import config_utils
import metadata
import logs

def run_stage5(setup):
    """Main driver function to run stage 5: kernel_solution
    This stage finds the kernel solution and (optionally) subtracts the model
    image
    :param object setup : an instance of the ReductionSetup class. See reduction_control.py

    :return: [status, report, reduction_metadata], stage5 status, report, 
     metadata file
    :rtype: array_like
    """

    stage5_version = 'stage5 v0.1'

    log = logs.start_stage_log(setup.red_dir, 'stage5', version=stage5_version)
    log.info('Setup:\n' + setup.summary() + '\n')
    try:
        from umatrix_routine import umatrix_construction, umatrix_bvector_construction, bvector_construction
         
    except ImportError:
        log.info('Uncompiled cython code, please run setup.py: e.g.\n python setup.py build_ext --inplace')
        status = 'KO'
        report = 'Uncompiled cython code, please run setup.py: e.g.\n python setup.py build_ext --inplace'
        return status, report

    # find the metadata
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')

    #determine kernel size based on maximum FWHM
    fwhm_max = 0.
    for stats_entry in reduction_metadata.images_stats[1]:
        if float(stats_entry['FWHM_X'])> fwhm_max:
            fwhm_max = stats_entry['FWHM_X']
        if float(stats_entry['FWHM_Y'])> fwhm_max:
            fwhm_max = stats_entry['FWHM_Y']

    sigma_max = fwhm_max*2.*(2.*np.log(2.))**0.5
    kernel_size = int(3.*float(reduction_metadata.reduction_parameters[1]['KER_RAD'][0]) * fwhm_max)


    if kernel_size:
        if kernel_size % 2 == 0:
            kernel_size = kernel_size + 1

    # find the images that need to be processed
    all_images = reduction_metadata.find_all_images(setup, reduction_metadata,
                                                    os.path.join(setup.red_dir, 'data'), log=log)

    new_images = reduction_metadata.find_images_need_to_be_process(setup, all_images,
                                                                   stage_number=5, rerun_all=None, log=log)

    kernel_directory_path = os.path.join(setup.red_dir, 'kernel')
    diffim_directory_path = os.path.join(setup.red_dir, 'diffim')
    if not os.path.exists(kernel_directory_path):
        os.mkdir(kernel_directory_path)
    if not os.path.exists(diffim_directory_path):
        os.mkdir(diffim_directory_path)
    reduction_metadata.update_column_to_layer('data_architecture', 'KERNEL_PATH', kernel_directory_path)
    # difference images are written for verbosity level > 0 
    reduction_metadata.update_column_to_layer('data_architecture', 'DIFFIM_PATH', diffim_directory_path)
    data_image_directory = reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]
    ref_directory_path = '.'
    #For a quick image subtraction, pre-calculate a sufficiently large u_matrix
    #based on the largest FWHM and store it to disk -> needs config switch

    try:
        reference_image_name = str(reduction_metadata.data_architecture[1]['REF_IMAGE'][0])
        reference_image_directory = str(reduction_metadata.data_architecture[1]['REF_PATH'][0])

        max_adu = 0.3*float(reduction_metadata.reduction_parameters[1]['MAXVAL'][0])

        logs.ifverbose(log, setup,'Using reference image:' + reference_image_name)
    except KeyError:
        log.ifverbose(log, setup,'Reference/Images ! Abort stage5')
        status = 'KO'
        report = 'No reference image found!'
        return status, report, reduction_metadata

    if not ('SHIFT_X' in reduction_metadata.images_stats[1].keys()) and (
            'SHIFT_Y' in reduction_metadata.images_stats[1].keys()):
        log.ifverbose(log, setup,'No xshift! run stage4 ! Abort stage5')
        status = 'KO'
        report = 'No alignment data found!'
        return status, report, reduction_metadata

    reference_image, bright_reference_mask = open_reference(setup, reference_image_directory, reference_image_name, kernel_size, max_adu, ref_extension = 0, log = log)
    #check if umatrix exists
    if os.path.exists(os.path.join(kernel_directory_path,'unweighted_u_matrix.npy')):
        umatrix, kernel_size_u, max_adu_restored = np.load(os.path.join(kernel_directory_path,'unweighted_u_matrix.npy'))
        if (kernel_size_u != kernel_size) or (max_adu_restored != max_adu):
            #calculate and store unweighted umatrix
            umatrix = umatrix_constant(reference_image, kernel_size)
            np.save(os.path.join(kernel_directory_path,'unweighted_u_matrix.npy'), [umatrix, kernel_size, max_adu])
            hdutmp = fits.PrimaryHDU(umatrix)
            hdutmp.writeto(os.path.join(kernel_directory_path,'unweighted_u_matrix.fits'),overwrite =True)
    else:
        #calculate and store unweighted umatrix 
        umatrix = umatrix_constant(reference_image, kernel_size)
        np.save(os.path.join(kernel_directory_path,'unweighted_u_matrix.npy'), [umatrix, kernel_size, max_adu])
        hdutmp = fits.PrimaryHDU(umatrix)
        hdutmp.writeto(os.path.join(kernel_directory_path,'unweighted_u_matrix.fits'),overwrite=True)

    if len(new_images) > 0:
        # find the reference image

        kernel_list = []
        for new_image in new_images:
            row_index = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == new_image)[0][0]
            x_shift, y_shift = -reduction_metadata.images_stats[1][row_index]['SHIFT_X'],-reduction_metadata.images_stats[1][row_index]['SHIFT_Y'] 
            try:
                data_image = open_data_image(setup, data_image_directory, new_image, bright_reference_mask, kernel_size, max_adu, xshift = x_shift, yshift = y_shift )
                missing_data_mask = (data_image == 0.)
                b_vector = bvector_constant(reference_image, data_image, kernel_size)
            	kernel_matrix, bkg_kernel, kernel_uncertainty  = kernel_solution(umatrix, b_vector, kernel_size)
              
                pscale = np.sum(kernel_matrix)
             
                np.save(os.path.join(kernel_directory_path,'kernel_'+new_image+'.npy'),kernel_matrix)
                hdu_kernel = fits.PrimaryHDU(kernel_matrix)
                hdu_kernel.writeto(os.path.join(kernel_directory_path,'kernel_'+new_image), overwrite = True)  
                hdu_kernel_err = fits.PrimaryHDU(kernel_uncertainty)
                hdu_kernel_err.writeto(os.path.join(kernel_directory_path,'kernel_err_'+new_image), overwrite = True)  

                logs.ifverbose(log, setup, 'b_vector calculated for:' + new_image)
                #CROP EDGE!
                difference_image = subtract_images(data_image, reference_image, kernel_matrix, kernel_size, bkg_kernel, missing_data_mask)
                difference_image_hdu = fits.PrimaryHDU(difference_image)
                difference_image_hdu.writeto(os.path.join(diffim_directory_path,'diff_'+new_image),overwrite = True)
            except:
                logs.ifverbose(log, setup,'kernel matrix computation or shift failed:' + new_image + '. skipping!')

            #append some metric for the kernel, perhaps its scale factor...
    reduction_metadata.update_reduction_metadata_reduction_status(new_images, stage_number=5, status=1, log=log)
    logs.close_log(log)
    status = 'OK'
    report = 'Completed successfully'

    return status, report

def noise_model(model_image, gain, readout_noise, flat=None, initialize=None):

    noise_image = np.copy(model_image)
    noise_image[noise_image == 0] = 1.
    noise_image = noise_image**2
    noise_image[noise_image != 1] = noise_image[noise_image != 1] + readout_noise * readout_noise
    weights = 1. / noise_image
    weights[noise_image == 1] = 0.
	
    return weights

def noise_model_constant(model_image):

    noise_image = np.copy(model_image)
    noise_image[noise_image == 0] = 1.
    weights = np.ones(np.shape(model_image))    
    weights[noise_image == 1] = 0.
	
    return weights

def noise_model_blurred_ref(reference_image,bright_mask,sigma_max):

    noise_image = np.copy(reference_image)
    good_region = np.copy(np.where(noise_image != 0))
    bad_region = np.copy(np.where(noise_image == 0))
    readout_noise = 12.

#    noise_image[noise_image == 0] = 1.
    noise_image = gaussian_filter(noise_image, sigma=sigma_max)
    noise_image = noise_image**2
    noise_image = noise_image + readout_noise * readout_noise
    weights = 1. / noise_image
    weights[bright_mask] = 0.
    return weights


def umatrix_constant(reference_image, ker_size, model_image=None, sigma_max = None, bright_mask = None):
    '''
    The kernel solution is supposed to implement the approach outlined in
    the Bramich 2008 paper. It generates the u matrix which is required
    for finding the best kernel and assumes it can be calculated
    sufficiently if the noise model either is neglected or similar on all
    model images. In order to run, it needs the largest possible kernel size
    and carefully masked regions which are expected to be affected on all
    data images.

    :param object image: reference image
    :param integer kernel size: edge length of the kernel in px

    :return: u matrix
    '''
    try:
        from umatrix_routine import umatrix_construction, umatrix_bvector_construction, bvector_construction
    except ImportError:
        print('cannot import cython module umatrix_routine')
        return []

    if ker_size:
        if ker_size % 2 == 0:
            kernel_size = ker_size + 1
        else:
            kernel_size = ker_size
    if model_image == None or sigma_max == None:
        weights = noise_model_constant(reference_image)
    else:
        weights = noise_model_blurred_ref(reference_image, bright_mask, sigma_max)
    # Prepare/initialize indices, vectors and matrices
    pandq = []
    n_kernel = kernel_size * kernel_size
    ncount = 0
    half_kernel_size = int(kernel_size) / 2
    for lidx in range(kernel_size):
        for midx in range(kernel_size):
            pandq.append((lidx - half_kernel_size, midx - half_kernel_size))

    u_matrix = umatrix_construction(reference_image, weights, pandq, n_kernel, kernel_size)

    return u_matrix


def bvector_constant(reference_image, data_image, ker_size, model_image=None, sigma_max = None, bright_mask = None):
    '''
    The kernel solution is supposed to implement the approach outlined in
    the Bramich 2008 paper. It generates the b_vector which is required
    for finding the best kernel and assumes it can be calculated
    sufficiently if the noise model either is neglected or similar on all
    model images. In order to run, it needs the largest possible kernel size
    and carefully masked regions on both - data and reference image

    :param object image: reference image
    :param integer kernel size: edge length of the kernel in px

    :return: b_vector
    '''
    try:
        from umatrix_routine import umatrix_construction, umatrix_bvector_construction, bvector_construction
    except ImportError:
        print('cannot import cython module umatrix_routine')
        return []

    if ker_size:
        if ker_size % 2 == 0:
            kernel_size = ker_size + 1
        else:
            kernel_size = ker_size
    if model_image == None or sigma_max == None:
        weights = noise_model_constant(reference_image)
    else:
        weights = noise_model_blurred_ref(reference_image, bright_mask, sigma_max)
    # Prepare/initialize indices, vectors and matrices
    pandq = []
    n_kernel = kernel_size * kernel_size
    ncount = 0
    half_kernel_size = int(kernel_size) / 2
    for lidx in range(kernel_size):
        for midx in range(kernel_size):
            pandq.append((lidx - half_kernel_size, midx - half_kernel_size))

    b_vector = bvector_construction(reference_image, data_image, weights, pandq, n_kernel, kernel_size)
    return b_vector

def kernel_preparation_matrix(data_image, reference_image, ker_size, model_image=None):
    '''
    The kernel solution is supposed to implement the approach outlined in
    the Bramich 2008 paper. The data image is indexed using i,j
    the reference image should have the same shape as the data image
    the kernel is indexed via l, m. kernel_size requires as input the edge
    length of the kernel. The kernel matrix k_lm and a background b0 
    k_lm, where l and m correspond to the kernel pixel indices defines 
    The resulting vector b is obtained from the matrix U_l,m,l

    :param object image: data image
    :param object image: reference image
    :param integer kernel size: edge length of the kernel in px
    :param string model: the image index of the astropy fits object

    :return: u matrix and b vector
    '''
    try:
        from umatrix_routine import umatrix_construction, umatrix_bvector_construction, bvector_construction
         
    except ImportError:
        log.info('Uncompiled cython code, please run setup.py: e.g.\n python setup.py build_ext --inplace')
        status = 'KO'
        report = 'Uncompiled cython code, please run setup.py: e.g.\n python setup.py build_ext --inplace'
        return status, report, reduction_metadata

    if np.shape(data_image) != np.shape(reference_image):
        return None

    if ker_size:
        if ker_size % 2 == 0:
            kernel_size = ker_size + 1
        else:
            kernel_size = ker_size

    if model_image == None:
        # assume that the model image is the reference...
        weights = noise_model(data_image, 1., 12., flat=None, initialize=True)
    else:
        weights = noise_model(model_image, 1., 12., flat=None, initialize=True)

    # Prepare/initialize indices, vectors and matrices
    pandq = []
    n_kernel = kernel_size * kernel_size
    ncount = 0
    half_kernel_size = int(kernel_size) / 2
    for lidx in range(kernel_size):
        for midx in range(kernel_size):
            pandq.append((lidx - half_kernel_size, midx - half_kernel_size))

    u_matrix, b_vector = umatrix_bvector_construction(reference_image, data_image,
                                                      weights, pandq, n_kernel, kernel_size)

    return u_matrix, b_vector

def kernel_solution(u_matrix, b_vector, kernel_size):
    '''
    reshape kernel solution for convolution and obtain uncertainty 
    from lstsq solution

    :param object array: u_matrix
	:param object array: b_vector
    :return: kernel matrix
    '''
    #For better stability: solve the least square problem via np.linalg.lstsq
    #inv_umatrix = np.linalg.inv(u_matrix)
    #a_vector = np.dot(inv_umatrix, b_vector)
    lstsq_result = np.linalg.lstsq(u_matrix,b_vector)
    a_vector = lstsq_result[0]
    mse = 1.#lstsq_result
    a_vector_err = np.diagonal(np.matrix(np.dot(u_matrix.T, u_matrix)).I)
    #MSE: mean square error of the residuals

    output_kernel = np.zeros(kernel_size * kernel_size, dtype=float)
    output_kernel = a_vector[:-1]
    output_kernel = output_kernel.reshape((kernel_size, kernel_size))

    err_kernel = np.zeros(kernel_size * kernel_size, dtype=float)
    err_kernel = a_vector_err[:-1]
    err_kernel = output_kernel.reshape((kernel_size, kernel_size))

    xyc = kernel_size / 2
    radius_square = (xyc)**2
    for idx in range(kernel_size):
        for jdx in range(kernel_size):
            if (idx - xyc)**2 + (jdx - xyc)**2 >= radius_square:
                output_kernel[idx, jdx] = 0.
                err_kernel[idx, jdx] = 0.

    output_kernel_2 = np.flip(np.flip(output_kernel, 0), 1)
    err_kernel_2 = np.flip(np.flip(err_kernel, 0), 1)

    return output_kernel_2, a_vector[-1], err_kernel_2

def subtract_images(data_image, reference_image, kernel, kernel_size, bkg_kernel, missing_mask):

    model_image = convolve2d(reference_image, kernel, mode='same')
    difference_image = model_image - data_image + bkg_kernel
    difference_image[missing_mask] = 0.
    difference_image = difference_image[kernel_size:-kernel_size,kernel_size:-kernel_size]
    return difference_image


def difference_image_single_iteration(ref_imagename, data_imagename, kernel_size,
                                      mask=None, max_adu=np.inf):
    '''
    Finding the difference image for a given refernce and data image
    for a defined kernel size without iteration or image subdivision

    :param object string: data image name
    :param object string: reference image name
    :param object integer kernel size: edge length of the kernel in px
    :param object image: bad pixel mask
    :param object integer: maximum allowed adu on images

    :return: u matrix and b vector
    '''

    ref_image, data_image, bright_mask = read_images(
        ref_imagename, data_imagename, kernel_size, max_adu)
    # construct U matrix
    n_kernel = kernel_size * kernel_size

    u_matrix, b_vector = naive_u_matrix(
        data_image, ref_image, kernel_size, model_image=None)
   
    output_kernel_2, bkg_kernel, err_kernel = kernel_solution(u_matrix, b_vector, kernel_size)
    model_image = convolve2d(ref_image, output_kernel_2, mode='same')

    difference_image = model_image - data_image + bkg_kernel
    difference_image[bright_mask] = 0.

    difference_image[kernel_size:-kernel_size, kernel_size:-kernel_size] = np.array(difference_image, float)
    return difference_image, output_kernel, bkg_kernel


def difference_image_subimages(ref_imagename, data_imagename,
    kernel_size, subimage_shape, overlap=0., mask=None, max_adu=np.inf):
    '''
    The overlap parameter follows the original DANDIA definition, i.e.
    it is applied to each dimension, i.e. 0.1 increases the subimage in
    x an y direction by (2*overlap + 1) * int(x_image_size/n_divisions)
    the subimage_element contains 4 entries: [x_divisions, y_divisions,
    x_selection, y_selection]
    '''

    ref_image, data_image, bright_mask = read_images_for_substamps(ref_imagename, data_imagename, kernel_size, max_adu)
    #allocate target image
    difference_image = np.zeros(np.shape(ref_image))

    #call via multiprocessing    
    subimages_args = []
    for idx in range(subimage_shape[0]):
        for jdx in range(subimage_shape[1]):
            print 'Solving for subimage ',[idx+1,jdx+1],' of ',subimage_shape
            x_shape, y_shape = np.shape(ref_image)
            x_subsize, y_subsize = x_shape/subimage_shape[0], y_shape/subimage_shape[1]
            subimage_element = subimage_shape+[idx,jdx]
             
            ref_image_substamp = ref_image[subimage_element[2] * x_subsize : (subimage_element[2] + 1) * x_subsize,
                                           subimage_element[3] * y_subsize : (subimage_element[3] + 1) * y_subsize]
            data_image_substamp = data_image[subimage_element[2] * x_subsize : (subimage_element[2] + 1) * x_subsize, 
                                             subimage_element[3] * y_subsize : (subimage_element[3] + 1) * y_subsize]
            bright_mask_substamp = bright_mask[subimage_element[2] * x_subsize : (subimage_element[2] + 1) * x_subsize,
                                               subimage_element[3] * y_subsize : (subimage_element[3] + 1) * y_subsize]
            
            # extend image size for convolution and kernel solution
            data_substamp_extended = np.zeros((np.shape(data_image_substamp)[0] + 2 * kernel_size,
            np.shape(data_image_substamp)[1] + 2 * kernel_size))
            ref_substamp_extended = np.zeros((np.shape(ref_image_substamp)[0] + 2 * kernel_size, 
            np.shape(ref_image_substamp)[1] + 2 * kernel_size))
            mask_substamp_extended = np.zeros((np.shape(ref_image_substamp)[0] + 2 * kernel_size,
            np.shape(ref_image_substamp)[1] + 2 * kernel_size))
            mask_substamp_extended = mask_substamp_extended >0

            data_substamp_extended[kernel_size:-kernel_size,
                                   kernel_size:-kernel_size] = np.array(data_image_substamp, float)
            ref_substamp_extended[kernel_size:-kernel_size,
                                  kernel_size:-kernel_size] = np.array(ref_image_substamp, float)
            mask_substamp_extended[kernel_size:-kernel_size,
                                   kernel_size:-kernel_size] = np.array(bright_mask_substamp, float)
            #difference_subimage = substamp_difference_image(ref_substamp_extended, data_substamp_extended,
            #                                                mask_substamp_extended, kernel_size, subimage_element, np.shape(ref_image))
            difference_image[subimage_element[2] * x_subsize : (subimage_element[2] + 1) * x_subsize,
                             subimage_element[3] * y_subsize : (subimage_element[3] + 1) * y_subsize] = difference_subimage


