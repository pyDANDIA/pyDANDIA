######################################################################
#
# stage4.py - Fourth stage of the pipeline. Align images to the reference
# image using autocorellation

#
# dependencies:
#      numpy 1.8+
#      astropy 1.0+
######################################################################

import numpy as np
import os
from astropy.io import fits
import sys

import config_utils

import metadata
import logs


def run_stage4(setup):
    """Main driver function to run stage 4: image alignement.
    This stage align the images to the reference frame!
    Input: setup - is an instance of the ReductionSetup class. See
           reduction_control.py
    Output: prepares the metadata file
    """

    stage4_version = 'stage4 v0.1'

    log = logs.start_stage_log(setup.red_dir, 'stage4', version=stage4_version)
    log.info('Setup:\n' + setup.summary() + '\n')

    # find the metadata
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')

    # find the images needed to treat
    all_images = reduction_metadata.find_all_images(setup, reduction_metadata,
                                                    os.path.join(setup.red_dir, 'data'), log=log)

    new_images = reduction_metadata.find_images_need_to_be_process(setup, all_images,
                                                                   stage_number=4, rerun_all=None, log=log)

    if len(new_images) > 0:

        # find the reference image
        try:
            reference_image_name = reduction_metadata.data_architecture[1]['REFERENCE_NAME']
            reference_image_directory = reduction_metadata.data_architecture[1]['IMAGES_PATH']
            reference_image = open_an_image(setup, reference_image_directory, reference_image_name, image_index=0,
                                            log=None)
            logs.ifverbose(log, setup,
                           'I found the reference frame:' + reference_image_name)
        except KeyError:
            logs.ifverbose(log, setup,
                           'I can not find any reference image! Aboard stage4')
            sys.exit(1)

        data = []
        for new_image in new_images:
            target_image = open_an_image(setup, reference_image_directory, new_image, image_index=0, log=None)

            try:
                x_new_center, y_new_center, x_shift, y_shift = find_x_y_shifts_from_the_reference_image(setup,
                                                                                                        reference_image,
                                                                                                        target_image,
                                                                                                        edgefraction=0.5,
                                                                                                        log=None)

                data.append([target_image, x_shift, y_shift])
                logs.ifverbose(log, setup,
                           'I found the image translation to the reference for frame:' +new_image)

            except:

                logs.ifverbose(log, setup,
                   'I can not find the image translation to the reference for frame:' + new_image+'. Aboard stage4!')
                sys.exit(1)


        if ('SHIFT_X' in reduction_metadata.images_stats[1].keys()) and (
                'SHIFT_Y' in reduction_metadata.images_stats[1].keys()):

            for index in range(len(data)):
                target_image = data[index][0]
                x_shift = data[index][1]
                y_shift = data[index][2]
                row_index = np.where(reduction_metadata.images_stats[1]['IMAGES'] == target_image)[0][0]
                reduction_metadata.update_a_cell_to_layer('images_stats', row_index, 'SHIFT_X', x_shift)
                reduction_metadata.update_a_cell_to_layer('images_stats', row_index, 'SHIFT_Y', y_shift)
                logs.ifverbose(log, setup,
                       'Updated metadata for image: '+target_image)
        else:
            logs.ifverbose(log, setup,
                           'I have to construct SHIFT_X and SHIFT_Y columns')

            sorted_data = np.copy(data).T

            for index in range(len(data)):
                target_image = data[index][0]

                row_index = np.where(reduction_metadata.images_stats[1]['IMAGES'] == target_image)[0][0]

                sorted_data[row_index] = data[index]

            column_format = 'int'
            column_unit = 'pix'

            reduction_metadata.add_column_to_layer('image_stats', 'SHIFT_X', sorted_data[:, 1],
                                                   new_column_format=column_format,
                                                   new_column_unit=column_unit)

            reduction_metadata.add_column_to_layer('image_stats', 'SHIFT_Y', sorted_data[:, 2],
                                                   new_column_format=column_format,
                                                   new_column_unit=column_unit)

    reduction_metadata.update_reduction_metadata_reduction_status(new_images, stage_number=4, status=1, log=log)

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
    '''
    Simply open an image using astropy.io.fits

    :param object reduction_metadata: the metadata object
    :param string image_directory: the image name
    :param string image_name: the image name
    :param string image_index: the image index of the astropy fits object

    :param boolean verbose: switch to True to have more informations

    :return: the opened image
    :rtype: astropy.image object
    '''
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


def find_x_y_shifts_from_the_reference_image(setup, reference_image, target_image, edgefraction=0.5, log=None):
    '''
    For a given reference image path pathref
    and a given image path pathimg
    the central part with an edge length of
    edgefraction times full edge length is used
    to correlate a revised central pixel position
    '''

    reference_shape = reference_image.shape
    if reference_shape != target_image.shape:
        logs.ifverbose(log, setup, 'The reference image and the target image dimensions does not match! Aboard stage4')
        sys.exit(1)

    x_center = reference_shape[0] / 2
    y_center = reference_shape[1] / 2

    half_x = int(edgefraction * float(reference_shape[0])) / 2
    half_y = int(edgefraction * float(reference_shape[1])) / 2

    reduce_template = reference_image[
                      x_center - half_x:x_center + half_x, y_center - half_y:y_center + half_y]

    reduce_image = target_image[
                   x_center - half_x:x_center + half_x, y_center - half_y:y_center + half_y]
    x_shift, y_shift = correlation_shift(reduce_template, reduce_image)

    x_new_center = -x_shift + x_center
    y_new_center = -y_shift + y_center

    return x_new_center, y_new_center, x_shift, y_shift


def correlation_shift(reference_image, target_image):
    ''' This function calculates the revised
    central pixel coordinate for imgdata based
    on a cross correlation with refdata
    '''

    reference_shape = reference_image.shape

    x_center = reference_shape[0] / 2
    y_center = reference_shape[1] / 2

    correlation = convolve_image_with_a_psf(np.matrix(reference_image),
                                            np.matrix(target_image), correlate=1)

    x_shift, y_shift = np.unravel_index(np.argmax(correlation), correlation.shape)

    good_shift_y = y_shift - y_center
    good_shift_x = x_shift - x_center
    return good_shift_y, good_shift_x


def convolve_image_with_a_psf(image, psf, fourrier_transform_psf=None, fourrier_transform_image=None,
                              correlate=None, auto_correlation=None):
    """
     NAME:
           CONVOLVE
     PURPOSE:
           Convolution of an image with a Point Spread Function (PSF)
     EXPLANATION:
           The default is to compute the convolution using a product of
           Fourier transforms (for speed).

     CALLING SEQUENCE:

           imconv = convolve( image1, psf, FT_PSF = psf_FT )
      or:
           correl = convolve( image1, image2, /CORREL )
      or:
           correl = convolve( image, /AUTO )

     INPUTS:
           image = 2-D np.array (matrix) to be convolved with psf
           psf = the Point Spread Function, (size < or = to size of image).

     OPTIONAL INPUT KEYWORDS:

           fourrier_transform_psf = passes out/in the Fourier transform of the PSF,
                   (so that it can be re-used the next time function is called).
           fourrier_transform_image= passes out/in the Fourier transform of image.

           correlate uses the np.conjugate of the Fourier transform of PSF,
                   to compute the cross-correlation of image and PSF,
                   (equivalent to IDL function convol() with NO rotation of PSF)

           auto_correlation computes the auto-correlation function of image using FFT.


     METHOD:
           When using FFT, PSF is centered & expanded to size of image.
     HISTORY:
           written, Frank Varosi, NASA/GSFC 1992.
           Appropriate precision type for result depending on input image
                                   Markus Hundertmark February 2006
           Fix the bug causing the recomputation of FFT(psf) and/or FFT(image)
                                   Sergey Koposov     December 2006
    """

    image_shape = np.array(image.shape)
    half_image_shape = image_shape / 2
    number_of_pixels = image.size

    if (fourrier_transform_image == None) or (fourrier_transform_image.ndim != 2):
        fourrier_transform_image = np.fft.ifft2(image)

    if (auto_correlation is not None):
        return np.roll(
            np.roll(number_of_pixels * np.real(
                np.fft.ifft2(fourrier_transform_image * np.conjugate(fourrier_transform_image))),
                    half_image_shape[0], 0), half_image_shape[1], 1)

    if (fourrier_transform_psf == None) or (
                fourrier_transform_psf.ndim != 2) or (fourrier_transform_psf.shape != image.shape) or (
                fourrier_transform_psf.dtype != image.dtype):
        psf_shape = np.array(psf.shape)

    location_maxima = np.maximum((half_image_shape - psf_shape / 2), 0)  # center PSF in new np.array,
    superior = np.maximum((psf_shape / 2 - half_image_shape), 0)  # handle all cases: smaller or bigger
    lower = np.minimum((s + image_shape - 1), (psf_shape - 1))

    fourrier_transform_psf = np.conjugate(image) * 0  # initialise with correct size+type according
    # to logic of conj and set values to 0 (type of ft_psf is conserved)
    fourrier_transform_psf[location_maxima[1]:location_maxima[1] + lower[1] - superior[1] + 1,
    location_maxima[0]:location_maxima[0] + lower[0] - superior[0] + 1] = psf[superior[1]:(lower[1]) + 1,
                                                                          superior[0]:(lower[0]) + 1]
    fourrier_transform_psf = np.fft.ifft2(fourrier_transform_psf)

    if (correlate is not None):
        convolution = number_of_pixels * np.real(
            np.fft.ifft2(fourrier_transform_image * np.conjugate(fourrier_transform_psf)))
    else:
        convolution = number_of_pixels * np.real(np.fft.ifft2(fourrier_transform_image * fourrier_transform_psf))

    half_image_shape += (image_shape % 2)  # shift correction for odd size images.

    return np.roll(np.roll(convolution, half_image_shape[0], 0), half_image_shape[1], 1)
