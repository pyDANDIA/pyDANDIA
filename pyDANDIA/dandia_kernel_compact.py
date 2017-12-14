#!/usr/bin/python

from __future__ import division
import numpy as np
import scipy as sp
from astropy.io import fits
import scipy.ndimage as ndimage
import time

def read_images():
    data_image = fits.open('data_image.fits')
    ref_image = fits.open('ref_image.fits')
    return ref_image[0].data, data_image[0].data + 100.0


# Depends on model_image

def noise_model(model_image, gain, readout_noise,  flat=None, initialize=None):
    if initialize == True:
        model_image = ndimage.gaussian_filter(np.copy(model_image),sigma=2,order=0)

    variance = np.copy(model_image)
    variance = variance / gain
    if flat != None:
        variance = variance / flat
        ron_term = readout_noise**2 / flat**2
    else:
        ron_term = readout_noise**2 * np.ones(np.shape(model_image))
    noise_image = variance + ron_term

    ##### CAREFUL HERE - CAN YOU BE SURE ALL NOISE VALUES ARE STRICTLY POSITIVE (i.e no zero or negative values?).
    ##### THIS IS IMPORTANT FOR WHEN YOU USE THEM IN THE MATRIX CALCULATIONS.
    ##### I RETURN THIS ARRAY AS INVERSE VARIANCES RATHER THAN VARIANCES, FORCING ANY ZERO OR NEGATIVE VARIANCES
    ##### TO BE ZERO IN THE INVERSE VARIANCE ARRAY, WHICH GIVES THEM ZERO WEIGHT IN THE MATRIX/VECTOR SUMS

    return noise_image


def naive_u_matrix(data_image, reference_image, ker_size, weights=None):
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

    # Check that reference image size equals target image size
    if np.shape(data_image) != np.shape(reference_image):
        return None
    refimxs, refimys = np.shape(reference_image)
    dataimxs, dataimys = np.shape(data_image)

    # Enforce an odd sized kernel so that the central kernel pixel can be placed at (u,v)=(0,0)
    if ker_size:
        if ker_size % 2 == 0:
            kernel_size = ker_size + 1
        else:
            kernel_size = ker_size

    # Calculate a half kernel size
    hkernel_size = int((kernel_size - 1)/2)

    # Cut away a border from the target image equal to the half-kernel size
    data_image_use = data_image[hkernel_size:(dataimxs - hkernel_size), hkernel_size:(dataimys - hkernel_size)]
    dataimxs_use = dataimxs - kernel_size + 1
    dataimys_use = dataimys - kernel_size + 1
    npix = dataimxs_use * dataimys_use

    # Initialise weight maps if required
    if weights == None:
        weights = noise_model(reference_image, 1., 0., flat=None, initialize=True)
        weights = 1. / weights[hkernel_size:(dataimxs - hkernel_size), hkernel_size:(dataimys - hkernel_size)]

    # Initialise kernel indices
    pandq = []
    n_kernel = kernel_size * kernel_size
    for lidx in range(kernel_size):
        for midx in range(kernel_size):
            pandq.append((lidx, midx))

    # Prepare the pattern array
    # - npix is number of pixels in the restricted image region (i.e. the number of data points)
    # - nkernel + 1 is the number of fitted parameters
    patt_arr = np.ones((npix, n_kernel + 1), dtype=float)
    for i in range(n_kernel):
        patt_arr[:,i] = np.reshape(reference_image[pandq[i][0]:(pandq[i][0] + dataimxs_use), pandq[i][1]:(pandq[i][1] + dataimys_use)], npix)

    # Prepare the weight array
    weight_arr = np.tile(np.reshape(weights, (npix, 1)), (1, n_kernel + 1))

    # Do some linear algebra
    u_matrix = np.dot(patt_arr.T, patt_arr*weight_arr)
    b_vector = np.dot(np.reshape(data_image_use*weights, (1, npix)), patt_arr).T

    # Return the u matrix and b vector
    return u_matrix, b_vector


if __name__ == '__main__':
    ref_image,data_image = read_images()

    kernel_size = 7

    print(np.shape(ref_image))
    print(np.shape(data_image))

    #construct U matrix

    t1 = time.clock()
    u_matrix, b_vector = naive_u_matrix(data_image, ref_image, kernel_size, weights=None)
    t2 = time.clock()

    print('done matrix')
    print(t2-t1)

    t1 = time.clock()
    u_L = sp.linalg.cholesky(u_matrix, lower=True)
    kernel_image = sp.linalg.cho_solve((u_L, True), b_vector)
    diff_background = kernel_image[-1]
    kernel_image = kernel_image[:-1]
    t2 = time.clock()

    print('done solution')
    print(t2-t1)

#    kernel_image = np.roll(kernel_image,int(kernel_image.size/2))

    #### Careful here - this line works, but only because of the way pandq are formed
    kernel_image = kernel_image.reshape((kernel_size,kernel_size))

    hl = fits.PrimaryHDU(np.abs(u_matrix))
    hl.writeto('tst_umatrix.fits',overwrite=True)

    hl2 = fits.PrimaryHDU(np.abs(kernel_image))
    hl2.writeto('kernel_image.fits',overwrite=True)

