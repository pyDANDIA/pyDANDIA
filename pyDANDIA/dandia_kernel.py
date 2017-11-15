from __future__ import division
import numpy as np
from numpy.linalg import inv
from astropy.io import fits
from umatrix_routine import umatrix_construction
import scipy.ndimage as ndimage


def read_images():
    data_image = fits.open('data_image.fits')
    ref_image = fits.open('ref_image.fits')
    return np.array(ref_image[0].data, float), np.array(data_image[1].data, float)

# Depends on model_image


def noise_model(model_image, gain, readout_noise,  flat=None, initialize=None):
    if initialize == True:
        model_image = ndimage.gaussian_filter(
            np.copy(model_image), sigma=2, order=0)

    variance = np.copy(model_image)
    variance = variance / gain
    if flat != None:
        variance = variance / flat
        ron_term = readout_noise**2 / flat**2
    else:
        ron_term = readout_noise**2 * np.ones(np.shape(model_image))
    noise_image = variance + ron_term

    return noise_image


def u_matrix_unimproved(data_image, reference_image, ker_size, weights=None):
    """
    The naive kernel solution is supposed to implement
    the Bramich 2008 implementation of the kernel solution as
    outlined in the original paper

    The data image is indexed using i,j
    the reference image should have the same shape as the data image
    the kernel is indexed via l,m and that is converted into 
    single 

    kernel_size requires as input the edge length of the kernel


    it consists of k_lm and a background b0 
    k_lm, where l and m correspond to the kernel pixel indices
    The resulting vector b is obtained from the matrix U_l,m,l
    In practice we are unrolling the index
    we assume a quadratic kernel
    """

    if np.shape(data_image) != np.shape(reference_image):
        return None

    if ker_size:
        if ker_size % 2 == 0:
            kernel_size = ker_size + 1
        else:
            kernel_size = ker_size

    if weights == None:
        weights = noise_model(reference_image, 1., 0.,
                              flat=None, initialize=True)
        weights = 1. / weights

    # Prepare/initialize indices, vectors and matrices
    pandq = []
    n_kernel = kernel_size * kernel_size
    for lidx in range(0, kernel_size):
        for midx in range(0, kernel_size):
            pandq.append((lidx, midx))

    u_matrix, b_vector = umatrix_construction(
        reference_image, data_image, weights, pandq, n_kernel, kernel_size)


    return u_matrix, b_vector


if __name__ == '__main__':
    ref_image, data_image = read_images()

    kernel_size = 7
    # construct U matrix
    u_matrix, b_vector = u_matrix_unimproved(
        data_image, ref_image, kernel_size, weights=None)

    kernel_image = np.dot(inv(u_matrix), b_vector)[:-1]
    kernel_image = np.roll(kernel_image[:-1], int(kernel_image.size / 2))
    kernel_image = kernel_image.reshape((kernel_size, kernel_size))

    hl = fits.PrimaryHDU(np.abs(u_matrix))
    hl.writeto('tst_umatrix.fits', overwrite=True)

    hl2 = fits.PrimaryHDU(np.abs(kernel_image))
    hl2.writeto('kernel_image.fits', overwrite=True)
