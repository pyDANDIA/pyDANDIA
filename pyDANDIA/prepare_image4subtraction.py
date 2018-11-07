import os
import numpy as np
from pyDANDIA import logs
from astropy.io import fits
from scipy.signal import convolve2d
from pyDANDIA.sky_background import mask_saturated_pixels_quick, generate_sky_model
from pyDANDIA.sky_background import fit_sky_background, generate_sky_model_image
from scipy.ndimage.interpolation import shift
from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground
from skimage.transform import resize
from ccdproc import cosmicray_lacosmic
import matplotlib.pyplot as plt

def background_mesh_median(image1, master_mask = []):
    image = np.copy(image1)
    zero_mask = (image == 0.)
    if master_mask != []:
        image[master_mask] = np.median(image1)
    #generate slices, iterate over centers
    box = min(int(np.shape(image)[0]//5), int(np.shape(image)[1]//5))
    centerx = int(box//2)
    centery = int(box//2)
    halfbox = int(box//2)
    image_shape = np.shape(image)
    xcen_range = range(centerx,image_shape[0],box) 
    ycen_range = range(centery,image_shape[1],box)
    percentile_bkg = np.zeros((len(xcen_range),len(ycen_range)))
    idx = 0
    jdx = 0
    for xcen in xcen_range:
        idx = 0
        for ycen in ycen_range:
            try:
                val =  np.percentile(image[xcen - halfbox:xcen + halfbox,ycen - halfbox:ycen+halfbox],perc)
                percentile_bkg[jdx,idx] = val
            except:
                percentile_bkg[jdx, idx] = 0
            idx += 1
        jdx += 1
    result = resize(percentile_bkg, np.shape(image),mode= 'symmetric',anti_aliasing = False)	
    result[zero_mask] =0.
    return result

def background_mesh(image):
    sigma_clip = SigmaClip(sigma=3., iters=10)
    bkg_estimator = MedianBackground()
    bkg = Background2D(image, (250, 250), filter_size=(3, 3),
        sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    return bkg.background

def circular_mask(radius,image):
    nx, ny = image.shape
    cx, cy = int(nx/2),int(ny/2)
    y,x = np.ogrid[-cx:nx-cx,-cy:ny-cy]
    mask = x*x + y*y > radius*radius
    return mask

def subtract_background(image_hdu, data_extension = 0):
    '''
    subtracting background from image_hdu
 
    :param object string: image_hdu
    :param object integer: data_extension
    :return: None
    '''

    zero_parts = data_image[data_extension].data == 0.
    image_hdu[data_extension].data = image_hdu[data_extension].data - background_mesh_median(image_hdu[data_extension].data)
    image_hdu[data_extension].data[zero_parts] = 0.

def extend_image_by_kernel(image_hdu, kernel_size, data_extension = 0):
    '''
    subtracting background from image_hdu
 
    :param object string: image_hdu
    :param object integer: data_extension
    :param object integer: kernel_size
    :return: extended_image
    '''

    data_extended = np.zeros((np.shape(image_hdu[data_extension].data)[0] + 2 * kernel_size, np.shape(image_hdu[data_extension].data)[1] + 2 * kernel_size))
    data_extended[kernel_size:-kernel_size, kernel_size:-kernel_size] = np.array(image_hdu[data_extension].data, float)
    
    return data_extended

def remove_cosmics(image_hdu, kernel_size, data_extension = 0):
    '''
    subtracting background from image_hdu
 
    :param object string: image_hdu
    :param object integer: data_extension
    :param object integer: kernel_size
    :return: image_replaced_cosmics
    '''
    ref_extended = cosmicray_lacosmic(ref_extended, sigclip=7, objlim = 7., satlevel = max_adu)[0]
    ref_image_unmasked = cosmicray_lacosmic(ref_image_unmasked, sigclip=7, objlim = 7, satlevel = max_adu)[0]
    ref_image_unmasked[bright_mask[kernel_size:-kernel_size,kernel_size:-kernel_size]] = 0. 


def construct_mask(image_hdu, kernel_size, mask_extension = 2, central_crop = None):
    #increase kernel size by 2 and define circular mask
    #Problem: introducing artefacts, too small for some cases ?! 
    kernel_size_plus = int(kernel_size) + int(kernel_size)
    mask_kernel = np.ones(kernel_size_plus * kernel_size_plus, dtype=float)
    mask_kernel = mask_kernel.reshape((kernel_size_plus, kernel_size_plus))
    xyc = int(kernel_size_plus / 2)
    radius_square = (xyc)**2
    for idx in range(kernel_size_plus):
        for jdx in range(kernel_size_plus):
            if (idx - xyc)**2 + (jdx - xyc)**2 >= radius_square:
                mask_kernel[idx, jdx] = 0.

    #instead of removing a maxshift edge, we try to just mask it
    if central_crop != None:
        tmp_mask = np.ones(np.shape(image_hdu[mask_extension].data))
        tmp_mask[central_crop:-central_crop,central_crop:-central_crop] = np.zeros(np.shape(image_hdu[data_extension].data[central_crop:-central_crop,central_crop:-central_crop]))
        mask_central = tmp_mask > 0
        image_hdu[mask_extension].data[mask_central] = 1

    mask_propagate = convolve2d(image_hdu[mask_extension].data, mask_kernel, mode='same')
    bright_mask = mask_propagate > 0.
    return reference_mask

def apply_integer_shift(image_hdu, xshift, yshift, data_extension = 0):
     '''
     apply integer shift based on cross-correlation, no image distortion

    :param object string: reference imagefilename
    :param object string: reference image filename
    :param object integer: kernel size edge length of the kernel in px
    :param object float: index of the maximum adu values
    :return: None
    '''
    img_shape = np.shape(image_hdu[data_extension].data)
    shifted = np.zeros(img_shape)

    if xshift>img_shape[0] or yshift>img_shape[1]:
        return []
    if xshift!=0 and yshift!=0:
        data_image[data_extension].data = shift(data_image[data_extension].data, (-yshift,-xshift), cval=0.)
    

def open_image(setup, data_image_directory, data_image_name, reference_mask = None, kernel_size,
                    max_adu, data_extension = 0, log = None, xshift = 0, yshift = 0, central_crop = None, subset = None, data_image1 = None, min_adu = None):
    '''
    reading difference image for constructing u matrix

    :param object : reference imagefilename
    :param object string: reference image filename
    :param object integer: kernel size edge length of the kernel in px
    :param object float: index of the maximum adu values
    :return: images, mask
    '''

    data_image = fits.open(os.path.join(data_image_directory, data_image_name), mmap=True)

    # image is a reference image, i.e. we need to update the mask
    if reference_mask == None:
        reference_mask = construct_mask(image_hdu, kernel_size, central_crop = None)
     

    data_image_unmasked = np.copy(data_image[data_extension].data)


    # extend image size for convolution and kernel solution
    data_extended = np.zeros((np.shape(data_image[data_extension].data)[0] + 2 * kernel_size, np.shape(data_image[data_extension].data)[1] + 2 * kernel_size))
    data_extended[kernel_size:-kernel_size, kernel_size:-
                 kernel_size] = np.array(data_image[data_extension].data, float)
    
    #replace saturated pixels with random noise or 0, bkg_sigma:
    #bkg_sigma = np.std(data_image_unmasked < img50pc) / (1.-2./np.pi)**0.5
    #apply consistent mask    
    data_image_unmasked[reference_mask[kernel_size:-kernel_size,kernel_size:-kernel_size]] = 0.#np.random.randn(len(data_image_unmasked[reference_mask[kernel_size:-kernel_size,kernel_size:-kernel_size]]))*bkg_sigma
    data_extended[reference_mask] = 0.

    return data_extended, data_image_unmasked

def open_reference(setup, ref_image_directory, ref_image_name, kernel_size, max_adu, ref_extension = 0, log = None, central_crop = None, subset = None, ref_image1 = None, min_adu = None, master_mask = []):
    '''
    reading difference image for constructing u matrix

    :param object string: reference imagefilename
    :param object string: reference image filename
    :param object integer: kernel size edge length of the kernel in px
    :param object float: index of the maximum adu values
    :return: images, mask
    '''


    
	
    img_shape = np.shape(ref_image[ref_extension].data) 
    ref50pc = np.median(ref_image[ref_extension].data)
    if master_mask != []:
        ref_image[ref_extension].data[master_mask] = max_adu + ref50pc + 1.

    ref_bright_mask_1 = (ref_image[ref_extension].data > max_adu + ref50pc)       
    ref_image[ref_extension].data = ref_image[ref_extension].data - background_mesh_perc(ref_image[ref_extension].data)

    ref_image_unmasked = np.copy(ref_image[ref_extension].data)
    if central_crop != None:
        tmp_image = np.zeros(np.shape(ref_image[ref_extension].data))
        tmp_image[central_crop:-central_crop,central_crop:-central_crop] = ref_image[ref_extension].data[central_crop:-central_crop,central_crop:-central_crop]
        ref_image[ref_extension].data = tmp_image

    mask_extended = np.zeros((np.shape(ref_image[ref_extension].data)[0] + 2 * kernel_size,
                             np.shape(ref_image[ref_extension].data)[1] + 2 * kernel_size))
    mask_extended[kernel_size:-kernel_size, kernel_size:-kernel_size][ref_bright_mask_1] = 1.

    ref_extended = np.zeros((np.shape(ref_image[ref_extension].data)[0] + 2 * kernel_size,
                             np.shape(ref_image[ref_extension].data)[1] + 2 * kernel_size))
    ref_extended[kernel_size:-kernel_size, kernel_size:-
                 kernel_size] = np.array(ref_image[ref_extension].data, float)
    
    #apply consistent mask
    ref_bright_mask = mask_extended > 0.
    mask_propagate = np.zeros(np.shape(ref_extended))
    mask_propagate[ref_bright_mask] = 1.
    #increase mask size to kernel size

      
    #replace saturated pixels with random noise or zero:
    bkg_sigma = np.std(ref_image_unmasked < np.median(ref_image_unmasked)) / (1.-2./np.pi)**0.5
    #apply consistent mask    
    ref_extended = cosmicray_lacosmic(ref_extended, sigclip=7, objlim = 7., satlevel = max_adu)[0]
    ref_image_unmasked = cosmicray_lacosmic(ref_image_unmasked, sigclip=7, objlim = 7, satlevel = max_adu)[0]
    ref_image_unmasked[bright_mask[kernel_size:-kernel_size,kernel_size:-kernel_size]] = 0. # np.random.randn(len(ref_image_unmasked[bright_mask[kernel_size:-kernel_size,kernel_size:-kernel_size]]))*bkg_sigma
 
 
    return np.array(ref_extended,dtype = float), bright_mask, np.array(ref_image_unmasked, dtype=float)
  

