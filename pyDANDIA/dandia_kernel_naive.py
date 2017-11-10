from __future__ import division
import numpy as np
from numpy.linalg import inv
from astropy.io import fits
import scipy.ndimage as ndimage

def read_images():
    data_image = fits.open('data_image.fits')
    ref_image = fits.open('ref_image.fits')
    return ref_image[0].data,data_image[1].data

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

    if np.shape(data_image) != np.shape(reference_image):
        return None

    if ker_size:
        if ker_size % 2 == 0:
            kernel_size = ker_size + 1
        else:
            kernel_size = ker_size

    if weights == None:
        weights = noise_model(reference_image, 1., 0., flat=None, initialize=True)
        weights = 1. / weights

    
    #Prepare/initialize indices, vectors and matrices
    pandq = []
    n_kernel = kernel_size * kernel_size
    for lidx in range(0, kernel_size):
        for midx in range(0, kernel_size):
            pandq.append((lidx, midx))
    b_vector = np.zeros(n_kernel + 1, dtype=float)
    u_matrix = np.zeros((n_kernel + 1, n_kernel + 1))
    ni_image, nj_image = np.shape(data_image)

    # main matrix - squared reference pixels...
    # naive for loop implementation preped for cython et al.
    for idx_p in range(n_kernel):
        for idx_q in range(idx_p,n_kernel):
            idx_l, idx_m = pandq[idx_p]
            idx_l_prime, idx_m_prime = pandq[idx_q]
            for idx_i in range(ni_image):
                for idx_j in range(nj_image):
                    
                    #check for index limit
                    if idx_i + idx_l < ni_image and  idx_j + idx_m < nj_image and idx_i + idx_l_prime < ni_image and  idx_j + idx_m_prime < nj_image :
                    
                        u_matrix[idx_p, idx_q] += reference_image[idx_i + idx_l, idx_j + idx_m] * \
                            reference_image[idx_i + idx_l_prime,
                                            idx_j + idx_m_prime] * weights[idx_i, idx_j]
                u_matrix[idx_q,idx_p] = u_matrix[idx_p, idx_q]

    # upper edge ( reference pixels / variance )

    for idx_p in [n_kernel]:
        for idx_q in range(n_kernel):
            idx_l, idx_m = kernel_size, kernel_size
            idx_l_prime, idx_m_prime = pandq[idx_q]
            for idx_i in range(ni_image):
                for idx_j in range(nj_image):
                    if idx_i + idx_l_prime < ni_image  and  idx_j + idx_m_prime < nj_image  :
                        u_matrix[idx_p, idx_q] += reference_image[idx_i + idx_l_prime, idx_j + idx_m_prime] * weights[idx_i, idx_j]

    # lower edge ( reference pixels / variance )

    for idx_p in range(n_kernel-1):
        for idx_q in [n_kernel]:
            idx_l, idx_m = pandq[idx_p]
            idx_l_prime, idx_m_prime = kernel_size,kernel_size
            for idx_i in range(ni_image):
                for idx_j in range(nj_image):
                    if idx_i + idx_l < ni_image  and  idx_j + idx_m < nj_image :
                        u_matrix[idx_p, idx_q] += reference_image[idx_i + idx_l, idx_j + idx_m] * weights[idx_i, idx_j]

    # final entry (1px) for background/noise
    u_matrix[n_kernel , n_kernel ] = np.sum(weights)

    # b vector
    for idx_p in range(n_kernel):
        idx_l, idx_m = pandq[idx_p]
        for idx_i in range(ni_image):
            for idx_j in range(nj_image):
                if idx_i + idx_l < ni_image  and  idx_j + idx_m < nj_image :
                    b_vector[idx_p] += data_image[idx_i, idx_j] * reference_image[idx_i + idx_l, idx_j + idx_m] * weights[idx_i, idx_j]

    # last entry for background
    b_vector[-1] = np.sum(data_image * weights)

    return u_matrix,b_vector

if __name__ == '__main__':
    ref_image,data_image = read_images()

    kernel_size = 7
    #construct U matrix
    u_matrix,b_vector = naive_u_matrix(data_image, ref_image, kernel_size, weights=None)

    kernel_image = np.dot(inv(u_matrix),b_vector)[:-1]
    kernel_image = np.roll(kernel_image[:-1],int(kernel_image.size/2))
    kernel_image = kernel_image.reshape((kernel_size,kernel_size))
   
    hl = fits.PrimaryHDU(np.abs(u_matrix))
    hl.writeto('tst_umatrix.fits',overwrite=True)
   
    hl2 = fits.PrimaryHDU(np.abs(kernel_image))
    hl2.writeto('kernel_image.fits',overwrite=True)

