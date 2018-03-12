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

import numpy as np
import os
from astropy.io import fits
from umatrix_routine import umatrix_construction
from scipy.signal import convolve2d
import sys

import config_utils

import metadata
import logs


def run_stage5(setup):
    """Main driver function to run stage 5: kernel_solution
    This stage finds the kernel solution and (optionally) subtracts the model
    image!
    :param object setup : an instance of the ReductionSetup class. See reduction_control.py

    :return: [status, report, reduction_metadata, umatrix, kernel], stage5 status, report, 
     metadata file, u_matrix (updated?), kernel (single or grid)
    :rtype: array_like

    """

    stage5_version = 'stage5 v0.1'

    log = logs.start_stage_log(setup.red_dir, 'stage5', version=stage5_version)
    log.info('Setup:\n' + setup.summary() + '\n')

    # find the metadata
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')var

    # find the images needed to treat
    all_images = reduction_metadata.find_all_images(setup, reduction_metadata,
                                                    os.path.join(setup.red_dir, 'data'), log=log)

    new_images = reduction_metadata.find_images_need_to_be_process(setup, all_images,
                                                                   stage_number=5, rerun_all=None, log=log)

    #For a quick image subtraction, pre-calculate a sufficiently large u_matrix
    #based on the largest FWHM and store it to disk -> need config switch
    if len(new_images) > 0:

        # find the reference image
        try:
            reference_image_name = reduction_metadata.data_architecture[1]['REFERENCE_NAME']
            reference_image_directory = reduction_metadata.data_architecture[1]['IMAGES_PATH']
            reference_image = open_an_image(setup, reference_image_directory, reference_image_name, image_index=0,
                                            log=None)
            logs.ifverbose(log, setup,
                           'Using reference image:' + reference_image_name)
        except KeyError:
            logs.ifverbose(log, setup,
                           'I can not find any reference image! Abort stage5')

            status = 'KO'
            report = 'No reference image found!'

            return status, report, reduction_metadata

        kernel_list = []
        for new_image in new_images:
            target_image = open_an_image(setup, reference_image_directory, new_image, image_index=0, log=None)

            try:
                u_matrix, b_vector = kernel_preparation_matrix(data_image, reference_image, ker_size, model_image=None)
            	kernel_matrix  = kernel_solution(u_matrix, b_vector)

                data.append([target_image, x_shift, y_shift])
                logs.ifverbose(log, setup,
                               'u_matrix and b_vector for image can be calculated:' + new_image)

            except:

                logs.ifverbose(log, setup,
                               'kernel matrix computation failed:' + new_image + '. Abort stage5!')

                status = 'KO'
                report = 'Kernel can not be determined for image:' + new_image + ' !'

                return status, report, reduction_metadata

        if ('KERNEL' in reduction_metadata.images_stats[1].keys())):

            for index in range(len(kernel_list)):
                
                logs.ifverbose(log, setup,
                               'Updated kernel metadata for image: ' + target_image)
        else:
            logs.ifverbose(log, setup,
                           'Introducing new kernel solution')

            #append some metric for the kernel, perhaps its scale factor...
            for index in range(len(data)):
                target_image = data[index][0]
                row_index = np.where(reduction_metadata.images_stats[1]['IMAGES'] == target_image)[0][0]

    reduction_metadata.update_reduction_metadata_reduction_status(new_images, stage_number=5, status=1, log=log)

    reduction_metadata.save_updated_metadata(
        reduction_metadata.data_architecture[1]['OUTPUT_DIRECTORY'][0],
        reduction_metadata.data_architecture[1]['METADATA_NAME'][0],
        log=log)

    logs.close_log(log)

    status = 'OK'
    report = 'Completed successfully'

    return status, report, reduction_metadata

def open_an_image(setup, image_directory, image_name,
                  image_index=0, log=None):
   
    image_directory_path = image_directory

    logs.ifverbose(log, setup,
                   'Attempting to open image ' + os.path.join(image_directory_path, image_name))

    try:

        image_data = fits.open(os.path.join(image_directory_path, image_name),
                               mmap=True)
        image_data = image_data[image_index]

        logs.ifverbose(log, setup, image_name + ' open : OK')

        return image_data

    except:
        logs.ifverbose(log, setup, image_name + ' open : not OK!')

        return None

def read_images(ref_image_filename, data_image_filename, kernel_size, max_adu):
	#to be updated with open_an_image ....
    '''
    The kernel solution is supposed to implement the approach outlined in
    the Bramich 2008 paper. The data image is indexed using i,j
    the reference image should have the same shape as the data image
    the kernel is indexed via l, m. kernel_size requires as input the edge
    length of the kernel. The kernel matrix k_lm and a background b0 
    k_lm, where l and m correspond to the kernel pixel indices defines 
    The resulting vector b is obtained from the matrix U_l,m,l

    :param object string: reference image filename
    :param object string: data image filename
    :param object integer: kernel size edge length of the kernel in px
    :param object float: index of the maximum adu values

    :return: image
    :rtype: ??
    '''

    data_image = fits.open(data_image_filename)
    ref_image = fits.open(ref_image_filename)
    kernel_size_plus = kernel_size + 2
    mask_kernel = np.ones(kernel_size_plus * kernel_size_plus, dtype=float)
    mask_kernel = mask_kernel.reshape((kernel_size_plus, kernel_size_plus))
    xyc = kernel_size_plus / 2
    radius_square = (xyc)**2
    for idx in range(kernel_size_plus):
        for jdx in range(kernel_size_plus):
            if (idx - xyc)**2 + (jdx - xyc)**2 >= radius_square:
                mask_kernel[idx, jdx] = 0.
    ref10pc = np.percentile(ref_image[0].data, 0.1)
    ref_image[0].data = ref_image[0].data - \
        np.percentile(ref_image[0].data, 0.1)

    # extend image size for convolution and kernel solution
    data_extended = np.zeros((np.shape(data_image[1].data)[
                             0] + 2 * kernel_size, np.shape(data_image[1].data)[1] + 2 * kernel_size))
    ref_extended = np.zeros((np.shape(ref_image[0].data)[
                            0] + 2 * kernel_size, np.shape(ref_image[0].data)[1] + 2 * kernel_size))
    data_extended[kernel_size:-kernel_size, kernel_size:-
                  kernel_size] = np.array(data_image[1].data, float)
    ref_extended[kernel_size:-kernel_size, kernel_size:-
                 kernel_size] = np.array(ref_image[0].data, float)

    ref_bright_mask = ref_extended > max_adu + ref10pc
    data_bright_mask = data_extended > max_adu
    mask_propagate = np.zeros(np.shape(data_extended))
    mask_propagate[ref_bright_mask] = 1.
    mask_propagate[data_bright_mask] = 1.
    mask_propagate = convolve2d(mask_propagate, mask_kernel, mode='same')
    bright_mask = mask_propagate > 0.
    ref_extended[bright_mask] = 0.
    data_extended[bright_mask] = 0.

    return ref_extended, data_extended, bright_mask

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
    :rtype: ??
    '''


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

    u_matrix, b_vector = umatrix_construction(
        reference_image, data_image, weights, pandq, n_kernel, kernel_size)

    # final entry (1px) for background/noise

    return u_matrix, b_vector

def kernel_solution(u_matrix, b_vector)
    '''
    reshape kernel solution for convolution

    :param object array: u_matrix
	:param object array: b_vector

    :return: kernel matrix
    :rtype: ??
    '''
    inv_umatrix = np.linalg.inv(u_matrix)
    a_vector = np.dot(inv_umatrix, b_vector)

#    kernel_no_back = kernel_image[0:len(kernel_image)-1]
    output_kernel = np.zeros(kernel_size * kernel_size, dtype=float)
    output_kernel = a_vector[:-1]
    output_kernel = output_kernel.reshape((kernel_size, kernel_size))
    xyc = kernel_size / 2
    radius_square = (xyc)**2
    for idx in range(kernel_size):
        for jdx in range(kernel_size):
            if (idx - xyc)**2 + (jdx - xyc)**2 >= radius_square:
                output_kernel[idx, jdx] = 0.
    output_kernel_2 = np.flip(np.flip(output_kernel, 0), 1)

	return output_kernel_2


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
    :rtype: ??
    '''

    ref_image, data_image, bright_mask = read_images(
        ref_imagename, data_imagename, kernel_size, max_adu)
    # construct U matrix
    n_kernel = kernel_size * kernel_size

    u_matrix, b_vector = naive_u_matrix(
        data_image, ref_image, kernel_size, model_image=None)
   
    output_kernel_2 = kernel_solution(u_matrix, b_vector)
    model_image = convolve2d(ref_image, output_kernel_2, mode='same')

    difference_image = model_image - data_image + a_vector[-1]
    difference_image[bright_mask] = 0.
    difference_image[-kernel_size - 2:, :] = 0.
    difference_image[0:kernel_size + 2, :] = 0.
    difference_image[:, -kernel_size - 2:] = 0.
    difference_image[:, 0:kernel_size + 2] = 0.

    return difference_image, output_kernel, a_vector[-1]



