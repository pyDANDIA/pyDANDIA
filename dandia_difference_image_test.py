#from __future__ import division
import numpy as np
from astropy.io import fits
from numpy.fft import ifft2
from numpy.fft import fft2
from umatrix_routine import umatrix_construction
from scipy.signal import convolve2d

def read_images_art(kernel_size):
    data_image = fits.open('art_data_image.fits')
    ref_image = fits.open('ref_image.fits')
    ref_image = ref_image[0].data - np.percentile(ref_image[0].data,0.1)
    #extend image size for convolution and kernel solution
    data_extended = np.zeros((np.shape(data_image[0].data)[0]+2*kernel_size,np.shape(data_image[0].data)[1]+2*kernel_size))
    ref_extended  = np.zeros((np.shape(ref_image[0].data)[0]+2*kernel_size,np.shape(ref_image[0].data)[1]+2*kernel_size))
    data_extended[kernel_size:-kernel_size,kernel_size:-kernel_size]=np.array(data_image[0].data,float)
    ref_extended[kernel_size:-kernel_size,kernel_size:-kernel_size]=np.array(ref_image[0].data,float)
    #NP.ARRAY REQUIRED FOR ALTERED BYTE ORDER (CYTHON CODE)
    return ref_extended,data_extended


def read_images(kernel_size):
    data_image = fits.open('data_image.fits')
    ref_image = fits.open('ref_image.fits')

    mask_kernel = np.ones(kernel_size*kernel_size,dtype=float)
    mask_kernel = mask_kernel.reshape((kernel_size, kernel_size))
    xyc = kernel_size/2
    radius_square = (xyc)**2
    for idx in range(kernel_size):
        for jdx in range(kernel_size):
            if (idx-xyc)**2+(jdx-xyc)**2>=radius_square:
                mask_kernel[idx,jdx]  = 0.
    ref10pc = np.percentile(ref_image[0].data,0.1)
    ref_image[0].data = ref_image[0].data - np.percentile(ref_image[0].data,0.1)

    #extend image size for convolution and kernel solution
    data_extended = np.zeros((np.shape(data_image[1].data)[0]+2*kernel_size,np.shape(data_image[1].data)[1]+2*kernel_size))
    ref_extended  = np.zeros((np.shape(ref_image[0].data)[0]+2*kernel_size,np.shape(ref_image[0].data)[1]+2*kernel_size))
    data_extended[kernel_size:-kernel_size,kernel_size:-kernel_size]=np.array(data_image[1].data,float)
    ref_extended[kernel_size:-kernel_size,kernel_size:-kernel_size]=np.array(ref_image[0].data,float)
    
    ref_bright_mask = ref_extended>30000.+ref10pc
    data_bright_mask = data_extended>30000.   
    mask_propagate = np.zeros(np.shape(data_extended))
    mask_propagate[ref_bright_mask] = 1.
    mask_propagate[data_bright_mask] = 1.
    mask_propagate = convolve2d(mask_propagate, mask_kernel,mode = 'same')
    bright_mask = mask_propagate>0.
    ref_extended[bright_mask]=0.
    data_extended[bright_mask]=0.


    #NP.ARRAY REQUIRED FOR ALTERED BYTE ORDER (CYTHON CODE)
    return ref_extended,data_extended,bright_mask

# Depends on model_image
def noise_model(model_image, gain, readout_noise, flat=None, initialize=None):
    #if initialize == True:
    #    model_image = ndimage.gaussian_filter(np.copy(model_image),sigma=2,order=0)

    noise_image = np.copy(model_image)
    #variance = variance / gain
    #if flat != None:
    #    variance = variance / flat
    #    ron_term = readout_noise**2 / flat**2
    #else:
    #    ron_term = readout_noise**2 * np.ones(np.shape(model_image))

    noise_image[noise_image==0]=1.
    noise_image = noise_image**2 + readout_noise**2
    #noise_extended  = np.ones(np.shape(model_image))
    weights = 1./noise_image
    weights[noise_image==1]=0.
    #hl = fits.PrimaryHDU(weights)
    #hl.writeto('tst_noise_naive.fits',overwrite=True)
    #return np.ones(np.shape(model_image))

    return weights
                
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
        #assume that the model image is the reference...
        weights = noise_model(data_image, 1., 0., flat=None, initialize = True)
     
    
    
    #Prepare/initialize indices, vectors and matrices
    pandq = []
    n_kernel = kernel_size * kernel_size
    ncount=0
    half_kernel_size = int(kernel_size)/2
    for lidx in range(kernel_size):
        for midx in range(kernel_size):
            pandq.append((lidx-half_kernel_size , midx-half_kernel_size))
            

    u_matrix, b_vector = umatrix_construction(reference_image, data_image, weights, pandq, n_kernel, kernel_size)

    # final entry (1px) for background/noise
    #hlm = fits.PrimaryHDU(np.abs(u_matrix))
    #hlm.writeto('tst_umatrix_naive2.fits',overwrite=True)

    #hlbb = fits.PrimaryHDU(b_vector)
    #hlbb.writeto('tst_bvector_naive2.fits',overwrite=True)
   
    return u_matrix,b_vector

if __name__ == '__main__':

    kernel_size = 19
    ref_image,data_image,bright_mask = read_images(kernel_size)

    #construct U matrix
    n_kernel = kernel_size * kernel_size

    u_matrix,b_vector = naive_u_matrix(data_image, ref_image, kernel_size, weights=None)
    inv_umatrix = np.linalg.inv(u_matrix)
    a_vector = np.dot(inv_umatrix, b_vector)

#    kernel_no_back = kernel_image[0:len(kernel_image)-1]
    output_kernel = np.zeros(kernel_size*kernel_size,dtype=float)
    output_kernel = a_vector[:-1]
    output_kernel = output_kernel.reshape((kernel_size, kernel_size))
    xyc = kernel_size/2
    radius_square = (xyc)**2
    for idx in range(kernel_size):
        for jdx in range(kernel_size):
            if (idx-xyc)**2+(jdx-xyc)**2>=radius_square:
                output_kernel[idx,jdx]  = 0.
    
    difference_image = convolve2d(ref_image, output_kernel,mode = 'same') - data_image 
    difference_image[bright_mask]=np.mean(difference_image)
    difference_image[-2*kernel_size:,:]=0.
    difference_image[0:2*kernel_size,:]=0.
    difference_image[:,-2*kernel_size:]=0.
    difference_image[:,0:2*kernel_size]=0.
    
    hl7 = fits.PrimaryHDU(difference_image)
    hl7.writeto('tst_dif18.fits',overwrite=True)


    hl5 = fits.PrimaryHDU(np.flip(np.flip(output_kernel,0),1))
    hl5.header['BKG'] = a_vector[-1]
    hl5.writeto('kernel_naive.fits',overwrite=True)

