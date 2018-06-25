from scipy.signal import convolve2d
from multiprocessing import Pool
import multiprocessing as mp

def subtract_images(data_image, reference_image, kernel, kernel_size, bkg_kernel, mask = None):
    model_image = convolve2d(reference_image, kernel, mode='same')
    difference_image = model_image - data_image + bkg_kernel
    if mask != None:
        difference_image[mask[kernel_size:-kernel_size,kernel_size:-kernel_size]] = 0.
    return difference_image


#def subtract_all_subimages(imagename):
   
#	    difference_image = subtract_images(data_image_unmasked, reference_stamps[substamp_idx][2], kernel_matrix, kernel_size, bkg_kernel)



