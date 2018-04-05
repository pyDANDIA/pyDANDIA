import os
import numpy as np
from pyDANDIA import logs
from astropy.io import fits
from scipy.signal import convolve2d
from scipy.ndimage.filters import gaussian_filter
from pyDANDIA.sky_background import mask_saturated_pixels, generate_sky_model
from pyDANDIA.sky_background import fit_sky_background, generate_sky_model_image
from scipy.ndimage.interpolation import shift

def background_subtract(setup, image, max_adu):
    masked_image = mask_saturated_pixels(setup, image, max_adu,log = None)
    sky_params = { 'background_type': 'gradient', 
          'nx': image.shape[1], 'ny': image.shape[0],
          'a0': 0.0, 'a1': 0.0, 'a2': 0.0 }
    sky_model = generate_sky_model(sky_params) 
    sky_fit = fit_sky_background(masked_image,sky_model,'gradient',log=None)
    sky_params['a0'] = sky_fit[0][0]
    sky_params['a1'] = sky_fit[0][1]
    sky_params['a2'] = sky_fit[0][2]
    #sky_model = generate_sky_model(sky_params)
    sky_model_image = generate_sky_model_image(sky_params)
    return image - sky_model_image

def read_images_for_substamps(ref_image_filename, data_image_filename, kernel_size, max_adu):
    data_image = fits.open(data_image_filename)
    ref_image = fits.open(ref_image_filename)
    # Masks
    kernel_size_plus = kernel_size + 2
    mask_kernel = np.ones(kernel_size_plus * kernel_size_plus, dtype=float)
    mask_kernel = mask_kernel.reshape((kernel_size_plus, kernel_size_plus))
    xyc = int(kernel_size_plus / 2)
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
    ref_complete = np.copy(ref_extended)
    ref_extended[bright_mask] = 0.
    data_extended[bright_mask] = 0.
    
    #half_kernel_mask
    

    return ref_extended, data_extended, bright_mask, ref_complete

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

def open_data_image(setup, data_image_directory, data_image_name, reference_mask, kernel_size,
                    max_adu, data_extension = 0, log = None, xshift = 0, yshift = 0, sigma_smooth = 0, central_crop = None):
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
    img50pc = np.median(data_image[data_extension].data)
    data_image[data_extension].data = background_subtract(setup, data_image[data_extension].data, img50pc)
    img_shape = np.shape(data_image[data_extension].data)
    shifted = np.zeros(img_shape)
    #smooth data image
    if sigma_smooth != 0:
        data_image[data_extension].data = gaussian_filter(data_image[data_extension].data, sigma=sigma_smooth)

    if xshift>img_shape[0] or yshift>img_shape[1]:
        return []
    data_image[data_extension].data = shift(data_image[data_extension].data, (-yshift,-xshift), cval=0.)
    data_image_unmasked = np.copy(data_image[data_extension].data)
    if central_crop != None:
        tmp_image = np.zeros(np.shape(data_image[data_extension].data))
        tmp_image[central_crop:-central_crop,central_crop:-central_crop] = data_image[data_extension].data[central_crop:-central_crop,central_crop:-central_crop]
        data_image[data_extension].data =tmp_image
    # extend image size for convolution and kernel solution
    data_extended = np.zeros((np.shape(data_image[data_extension].data)[0] + 2 * kernel_size, np.shape(data_image[data_extension].data)[1] + 2 * kernel_size))
    data_extended[kernel_size:-kernel_size, kernel_size:-
                 kernel_size] = np.array(data_image[data_extension].data, float)
    
    #apply consistent mask    
    data_extended[reference_mask] = 0.
    #dout = fits.PrimaryHDU(data_extended)
    #dout.writeto('datext'+data_image_name,overwrite=True)
    return data_extended, data_image_unmasked

def open_reference(setup, ref_image_directory, ref_image_name, kernel_size,
                   max_adu, ref_extension = 0, log = None, central_crop = None):
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
    kernel_size_plus = int(kernel_size)+4
    mask_kernel = np.ones(kernel_size_plus * kernel_size_plus, dtype=float)
    mask_kernel = mask_kernel.reshape((kernel_size_plus, kernel_size_plus))
    xyc = int(kernel_size_plus / 2)
    radius_square = (xyc)**2
    for idx in range(kernel_size_plus):
        for jdx in range(kernel_size_plus):
            if (idx - xyc)**2 + (jdx - xyc)**2 >= radius_square:
                mask_kernel[idx, jdx] = 0.
    img_shape = np.shape(ref_image[ref_extension].data) 
    ref50pc = np.median(ref_image[ref_extension].data)
    ref_bright_mask = ref_image[ref_extension].data > max_adu + ref50pc
    ref_image[ref_extension].data = background_subtract(setup, ref_image[ref_extension].data, ref50pc)

    ref_image_unmasked = np.copy(ref_image[ref_extension].data)
    if central_crop != None:
        tmp_image = np.zeros(np.shape(ref_image[ref_extension].data))
        tmp_image[central_crop:-central_crop,central_crop:-central_crop] = ref_image[ref_extension].data[central_crop:-central_crop,central_crop:-central_crop]
        ref_image[ref_extension].data = tmp_image

    mask_extended = np.zeros((np.shape(ref_image[ref_extension].data)[0] + 2 * kernel_size,
                             np.shape(ref_image[ref_extension].data)[1] + 2 * kernel_size))
    mask_extended[kernel_size:-kernel_size, kernel_size:-kernel_size][ref_bright_mask] = 1.
    ref_extended = np.zeros((np.shape(ref_image[ref_extension].data)[0] + 2 * kernel_size,
                             np.shape(ref_image[ref_extension].data)[1] + 2 * kernel_size))
    ref_extended[kernel_size:-kernel_size, kernel_size:-
                 kernel_size] = np.array(ref_image[ref_extension].data, float)
    
    #apply consistent mask
    ref_bright_mask = mask_extended > 0.
    mask_propagate = np.zeros(np.shape(ref_extended))
    mask_propagate[ref_bright_mask] = 1.
    #increase mask size to kernel size
    mask_propagate = convolve2d(mask_propagate, mask_kernel, mode='same')
    bright_mask = mask_propagate > 0.
    ref_extended[bright_mask] = 0.
    #dout = fits.PrimaryHDU(ref_extended)
    #dout.writeto('refext'+ref_image_name,overwrite=True)
    return ref_extended, bright_mask, ref_image_unmasked

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
    xyc = int(kernel_size_plus / 2)
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
