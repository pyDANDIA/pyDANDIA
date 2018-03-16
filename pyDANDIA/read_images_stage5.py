import os
import numpy as np
import logs
from astropy.io import fits
from scipy.signal import convolve2d

def read_images_for_substamps(ref_image_filename, data_image_filename, kernel_size, max_adu):
    data_image = fits.open(data_image_filename)
    ref_image = fits.open(ref_image_filename)
    # Masks
    kernel_size_plus = kernel_size
    mask_kernel = np.ones(kernel_size_plus * kernel_size_plus, dtype=float)
    mask_kernel = mask_kernel.reshape((kernel_size_plus, kernel_size_plus))
    xyc = kernel_size_plus / 2
    radius_square = (xyc)**2
    for idx in range(kernel_size_plus):
        for jdx in range(kernel_size_plus):
            if (idx - xyc)**2 + (jdx - xyc)**2 >= radius_square:
                mask_kernel[idx, jdx] = 0.
    ref10pc = np.percentile(ref_image[0].data, 10)
    ref_image[0].data = ref_image[0].data - \
        np.percentile(ref_image[0].data, 0.1)

    # extend image size for convolution and kernel solution
    data_extended = np.zeros((np.shape(data_image[1].data)))
    ref_extended = np.zeros((np.shape(ref_image[0].data)))
    data_extended = np.array(data_image[1].data, float)
    ref_extended = np.array(ref_image[0].data, float)

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

def open_data_image(setup, data_image_directory, data_image_name, reference_mask, kernel_size, max_adu, data_extension = 0, log = None):
    '''
    reading difference image for constructing u matrix

    :param object string: reference imagefilename
    :param object string: reference image filename
    :param object integer: kernel size edge length of the kernel in px
    :param object float: index of the maximum adu values
    :return: images, mask
    '''

    logs.ifverbose(log, setup, 'Attempting to open data image ' + os.path.join(data_image_directory, data_image_name))

    data_image = fits.open(os.path.join(data_image_directory, data_image_name), mmap=True)

	#increase kernel size by 2 and define circular mask
    kernel_size_plus = kernel_size + 2
    mask_kernel = np.ones(kernel_size_plus * kernel_size_plus, dtype=float)
    mask_kernel = mask_kernel.reshape((kernel_size_plus, kernel_size_plus))
    xyc = kernel_size_plus / 2
    radius_square = (xyc)**2
    for idx in range(kernel_size_plus):
        for jdx in range(kernel_size_plus):
            if (idx - xyc)**2 + (jdx - xyc)**2 >= radius_square:
                mask_kernel[idx, jdx] = 0.

    # extend image size for convolution and kernel solution
    data_extended = np.zeros((np.shape(data_image[data_extension].data)[0] + 2 * kernel_size, np.shape(data_image[data_extension].data)[1] + 2 * kernel_size))
    data_extended[kernel_size:-kernel_size, kernel_size:-
                 kernel_size] = np.array(data_image[data_extension].data, float)
    #apply consistent mask    
    data_extended[reference_mask] = 0.

    return data_extended

def open_reference(setup, ref_image_directory, ref_image_name, kernel_size,
                   max_adu, ref_extension = 0, log = None):
    '''
    reading difference image for constructing u matrix

    :param object string: reference imagefilename
    :param object string: reference image filename
    :param object integer: kernel size edge length of the kernel in px
    :param object float: index of the maximum adu values
    :return: images, mask
    '''

    logs.ifverbose(log, setup,
                   'Attempting to open ref image ' + os.path.join(ref_image_directory, ref_image_name))

    ref_image = fits.open(os.path.join(ref_image_directory, ref_image_name), mmap=True)
	#increase kernel size by 2 and define circular mask
    kernel_size_plus = kernel_size + 2
    print kernel_size_plus
    mask_kernel = np.ones(kernel_size_plus * kernel_size_plus, dtype=float)
    mask_kernel = mask_kernel.reshape((kernel_size_plus, kernel_size_plus))
    xyc = kernel_size_plus / 2
    radius_square = (xyc)**2
    for idx in range(kernel_size_plus):
        for jdx in range(kernel_size_plus):
            if (idx - xyc)**2 + (jdx - xyc)**2 >= radius_square:
                mask_kernel[idx, jdx] = 0.

    #subtract background estimate using 10% percentile
    ref10pc = np.percentile(ref_image[ref_extension].data, 10.)
    ref_image[ref_extension].data = ref_image[ref_extension].data - \
        np.percentile(ref_image[ref_extension].data, 10.)
    logs.ifverbose(log, setup,
                   'Background reference= ' + str(ref10pc))

    # extend image size for convolution and kernel solution
    ref_extended = np.zeros((np.shape(ref_image[ref_extension].data)[0] + 2 * kernel_size,
                             np.shape(ref_image[ref_extension].data)[1] + 2 * kernel_size))
    ref_extended[kernel_size:-kernel_size, kernel_size:-
                 kernel_size] = np.array(ref_image[ref_extension].data, float)
    
    #apply consistent mask
    ref_bright_mask = ref_extended > max_adu + ref10pc
    mask_propagate = np.zeros(np.shape(ref_extended))
    mask_propagate[ref_bright_mask] = 1.
    #increase mask size to kernel size
    mask_propagate = convolve2d(mask_propagate, mask_kernel, mode='same')
    bright_mask = mask_propagate > 0.
    ref_extended[bright_mask] = 0.

    return ref_extended, bright_mask

def open_images(setup, ref_image_directory, data_image_directory, ref_image_name,
                data_image_name, kernel_size, max_adu, ref_extension = 0, 
                data_image_extension = 0, log = None):
	#to be updated with open_an_image ....
    '''
    Reference and data image needs to be opened jointly and bright pixels
    are masked on both images depending on the corresponding kernel size
    and max_adu

    :param object string: reference imagefilename
    :param object string: data image filename
    :param object string: reference image filename
    :param object string: data image filename
    :param object integer: kernel size edge length of the kernel in px
    :param object float: index of the maximum adu values
    :return: images, mask
    '''

    logs.ifverbose(log, setup,
                   'Attempting to open data image ' + os.path.join(data_image_directory, data_image_name))

    data_image = fits.open(os.path.join(data_image_directory_path, data_image_name), mmap=True)

    logs.ifverbose(log, setup,
                   'Attempting to open ref image ' + os.path.join(ref_image_directory, ref_image_name))

    ref_image = fits.open(os.path.join(ref_image_directory_path, ref_image_name), mmap=True)

	#increase kernel size by 2 and define circular mask
    kernel_size_plus = kernel_size + 2
    mask_kernel = np.ones(kernel_size_plus * kernel_size_plus, dtype=float)
    mask_kernel = mask_kernel.reshape((kernel_size_plus, kernel_size_plus))
    xyc = kernel_size_plus / 2
    radius_square = (xyc)**2
    for idx in range(kernel_size_plus):
        for jdx in range(kernel_size_plus):
            if (idx - xyc)**2 + (jdx - xyc)**2 >= radius_square:
                mask_kernel[idx, jdx] = 0.

    #subtract background estimate using 10% percentile
    ref10pc = np.percentile(ref_image[ref_extension].data, 10.)
    ref_image[ref_extension].data = ref_image[ref_extension].data - \
        np.percentile(ref_image[ref_extension].data, 10.)

    logs.ifverbose(log, setup,
                   'Background reference= ' + str(ref10pc))

    # extend image size for convolution and kernel solution
    data_extended = np.zeros((np.shape(data_image[data_image_extension].data)[
                             0] + 2 * kernel_size, np.shape(data_image[data_image_extension].data)[1] + 2 * kernel_size))
    ref_extended = np.zeros((np.shape(ref_image[ref_image_extension].data)[
                            0] + 2 * kernel_size, np.shape(ref_image[ref_image_extension].data)[1] + 2 * kernel_size))
    data_extended[kernel_size:-kernel_size, kernel_size:-
                  kernel_size] = np.array(data_image[data_image_extension].data, float)
    ref_extended[kernel_size:-kernel_size, kernel_size:-
                 kernel_size] = np.array(ref_image[ref_image_extension].data, float)
    
    #apply consistent mask
    ref_bright_mask = ref_extended > max_adu + ref10pc
    data_bright_mask = data_extended > max_adu
    mask_propagate = np.zeros(np.shape(data_extended))
    mask_propagate[ref_bright_mask] = 1.
    mask_propagate[data_bright_mask] = 1.
    #increase mask size to kernel size
    mask_propagate = convolve2d(mask_propagate, mask_kernel, mode='same')
    bright_mask = mask_propagate > 0.
    ref_extended[bright_mask] = 0.
    data_extended[bright_mask] = 0.

    return ref_extended, data_extended, bright_mask
