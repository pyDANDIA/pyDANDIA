######################################################################
#
# convolution.py - Module defining the convolution methods.
# For model details see individual function descriptions.
#
# dependencies:
#      numpy 1.8+
######################################################################

import numpy as np



def convolve_image_with_a_psf(image, psf, fourrier_transform_psf=None, fourrier_transform_image=None,
                              correlate=None, auto_correlation=None):
    """
    Efficient convolution in Fourrier Space


    :param object image: the image data (i.e image.data)
    :param object psf: the kernel which gonna be convolve
    :param object fourrier_transform_psf: the kernel in fourrier space
    :param object fourrier_transform_image: the imagein fourrier space
    :param boolean correlate: ???
    :param boolean auto_correlation: ???


    :return: ??
    :rtype: ??
    """

    image_shape = np.array(image.shape)
    half_image_shape = (image_shape / 2).astype(int)
    number_of_pixels = image.size

    if (fourrier_transform_image == None) or (fourrier_transform_image.ndim != 2):
        fourrier_transform_image = np.fft.ifft2(image)

    if (auto_correlation is not None):
        return np.roll(
            np.roll(number_of_pixels * np.real(
                np.fft.fft2(fourrier_transform_image * np.conjugate(fourrier_transform_image))),
                    half_image_shape[0], 0), half_image_shape[1], 1)

    if (fourrier_transform_psf == None) or (
                fourrier_transform_psf.ndim != 2) or (fourrier_transform_psf.shape != image.shape) or (
                fourrier_transform_psf.dtype != image.dtype):
        psf_shape = np.array(psf.shape)

        location_maxima = np.maximum((half_image_shape - psf_shape / 2).astype(int), 0)  # center PSF in new np.array,
        superior = np.maximum((psf_shape / 2 - half_image_shape).astype(int), 0)  # handle all cases: smaller or bigger
        lower = np.minimum((superior + image_shape - 1).astype(int), (psf_shape - 1))

        fourrier_transform_psf = np.conjugate(image) * 0  # initialise with correct size+type according
        # to logic of conj and set values to 0 (type of ft_psf is conserved)
        fourrier_transform_psf[location_maxima[1]:location_maxima[1] + lower[1] - superior[1] + 1,
        location_maxima[0]:location_maxima[0] + lower[0] - superior[0] + 1] = psf[superior[1]:(lower[1]) + 1,
                                                                              superior[0]:(lower[0]) + 1]
        fourrier_transform_psf = np.fft.ifft2(fourrier_transform_psf)

    if (correlate is not None):
        convolution = number_of_pixels * np.real(
            np.fft.fft2(fourrier_transform_image * np.conjugate(fourrier_transform_psf)))
    else:
        convolution = number_of_pixels * np.real(np.fft.fft2(fourrier_transform_image * fourrier_transform_psf))

    half_image_shape += (image_shape % 2)  # shift correction for odd size images.

    return np.roll(np.roll(convolution, half_image_shape[0], 0), half_image_shape[1], 1)