import os
import numpy as np
from pyDANDIA import logs
from astropy.io import fits
from scipy.signal import convolve2d
from scipy import ndimage
from scipy import fftpack 
from scipy.ndimage.filters import gaussian_filter
from pyDANDIA.sky_background import mask_saturated_pixels_quick, generate_sky_model
from pyDANDIA.sky_background import fit_sky_background, generate_sky_model_image
from scipy.ndimage.interpolation import shift
from astropy.stats import SigmaClip
from photutils import Background2D, MedianBackground
from skimage.transform import resize
from ccdproc import cosmicray_lacosmic
import matplotlib.pyplot as plt
from pyDANDIA import psf

def background_mesh_perc(image1,perc=30,box_guess=300, master_mask = []):

    image = np.copy(image1)
    zero_mask = (image == 0.)
    if master_mask != []:
        image[master_mask] = np.median(image1)
    #generate slices, iterate over centers
    if box_guess > int(np.shape(image)[0]/5) and box_guess > int(np.shape(image)[1]/5):
        box = min(int(np.shape(image)[0]/5), int(np.shape(image)[1]/5))
    else:
        box = box_guess
    centerx = int(box/2)
    centery = int(box/2)
    halfbox = int(box/2)
    image_shape = np.shape(image)
    xcen_range = range(centerx,image_shape[0],box) 
    ycen_range = range(centery,image_shape[1],box)
    percentile_bkg = np.zeros((len(xcen_range),len(ycen_range)))
    idx = 0
    jdx = 0
    perc5 = np.percentile(image,5)
    for xcen in xcen_range:
        idx = 0
        for ycen in ycen_range:
            try:
                positive = image[xcen - halfbox:xcen + halfbox,ycen - halfbox:ycen+halfbox] > perc5
                val =  np.percentile(image[xcen - halfbox:xcen + halfbox,ycen - halfbox:ycen+halfbox][positive],perc)

                percentile_bkg[jdx,idx] = val
            except:
                percentile_bkg[jdx, idx] = 0
            idx += 1
        jdx += 1

    result = resize(percentile_bkg, np.shape(image),mode= 'symmetric')	
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

def background_subtract(setup, image, max_adu, min_adu=None):
    masked_image = mask_saturated_pixels_quick(setup, image, max_adu, min_value = min_adu, log = None)
    
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

def open_data_image(setup, data_image_directory, data_image_name, reference_mask, kernel_size,
                    max_adu, data_extension = 0, log = None, xshift = 0, yshift = 0, sigma_smooth = 0, central_crop = None, subset = None, data_image1 = None, min_adu = None):
    '''
    reading difference image for constructing u matrix

    :param object string: reference imagefilename
    :param object string: reference image filename
    :param object integer: kernel size edge length of the kernel in px
    :param object float: index of the maximum adu values
    :return: images, mask
    '''
    #replacing the bkg with noise
    np.random.seed(0)

    #data_image1 is a large format image for processing subimages
    if data_image1 == None:
        data_image = fits.open(os.path.join(data_image_directory, data_image_name), mmap=True)
    if subset != None and data_image1 == None:
        #crop subimage
        data_image[data_extension].data=data_image[data_extension].data[subset[0]:subset[1],subset[2]:subset[3]]
    if subset != None and data_image1 != None:
        data_image = fits.HDUList(fits.PrimaryHDU(data_image1[data_extension].data[subset[0]:subset[1],subset[2]:subset[3]]))

    img50pc = np.median(data_image[data_extension].data)
    zero_parts = data_image[data_extension].data == 0.
    #import pdb;
    #pdb.set_trace()
    data_image[data_extension].data = data_image[data_extension].data - background_mesh_perc(data_image[data_extension].data)
    #mask =  (data_image[data_extension].data < np.percentile(data_image[data_extension].data.ravel(), 95)) & (data_image[data_extension].data!=0)


    #yfit,xfit=np.indices(data_image[0].data.shape)
    #res = psf.fit_background(data_image[0].data,yfit,xfit,mask,background_model='Gradient')
    ##bb = psf.GradientBackground()
    #momo=bb.background_model(yfit,xfit,res[0])
    #data_image[data_extension].data = data_image[data_extension].data-momo
    data_image[data_extension].data[zero_parts] = 0.
    #import pdb;
    #pdb.set_trace()
    img_shape = np.shape(data_image[data_extension].data)
    shifted = np.zeros(img_shape)
    #smooth data image
    if sigma_smooth != 0:
        data_image[data_extension].data = gaussian_filter(data_image[data_extension].data, sigma=sigma_smooth)

    if xshift>img_shape[0] or yshift>img_shape[1]:
        return []
    if xshift!=0 and yshift!=0:
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
    
    #replace saturated pixels with random noise or 0, bkg_sigma:
    #bkg_sigma = np.std(data_image_unmasked < img50pc) / (1.-2./np.pi)**0.5
    #apply consistent mask    
    data_image_unmasked[reference_mask[kernel_size:-kernel_size,kernel_size:-kernel_size]] = 0.#np.random.randn(len(data_image_unmasked[reference_mask[kernel_size:-kernel_size,kernel_size:-kernel_size]]))*bkg_sigma
    data_extended[reference_mask] = 0.



    return data_extended, data_image_unmasked

def open_reference(setup, ref_image_directory, ref_image_name, kernel_size, max_adu, ref_extension = 0, log = None, central_crop = None, subset = None, ref_image1 = None, min_adu = None, master_mask = [], external_weight = None):
    '''
    reading difference image for constructing u matrix

    :param object string: reference imagefilename
    :param object string: reference image filename
    :param object integer: kernel size edge length of the kernel in px
    :param object float: index of the maximum adu values
    :return: images, mask
    '''


    if ref_image1 == None:
        ref_image = fits.open(os.path.join(ref_image_directory, ref_image_name), mmap=True)
    if subset != None and ref_image1 == None:
        ref_image[ref_extension].data=ref_image[ref_extension].data[subset[0]:subset[1],subset[2]:subset[3]]
    if subset != None and ref_image1 != None:
        ref_image = fits.HDUList(fits.PrimaryHDU(ref_image1[ref_extension].data[subset[0]:subset[1],subset[2]:subset[3]]))
    
	#increase kernel size by 1.5 and define circular mask
    kernel_size_plus = int(kernel_size) + int(kernel_size)
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
    if master_mask != []:
        ref_image[ref_extension].data[master_mask] = max_adu + ref50pc + 1.
    
    ref_bright_mask_1 = (ref_image[ref_extension].data > max_adu + ref50pc)

    bkg_image = background_mesh_perc(ref_image[ref_extension].data)
    if external_weight is not None:
        try:
            noise_image = external_weight + np.copy(ref_image[ref_extension].data) 
	except:
	    noise_image = np.zeros(np.shape(ref_image[ref_extension].data))
            print('format mismatch (noise model construction)')
    else:
        noise_image = np.copy(ref_image[ref_extension].data) 
    
    #noise_image = gaussian_filter(noise_image, sigma=kernel_size/2)
    ref_image[ref_extension].data = ref_image[ref_extension].data - bkg_image

    #mask = (ref_image[ref_extension].data<np.percentile(ref_image[ref_extension].data.ravel(),95)) & (ref_image[ref_extension].data!=0)
    #yfit, xfit = np.indices(ref_image[0].data.shape)
    #res = psf.fit_background(ref_image[0].data, yfit, xfit, mask, background_model='Gradient')
    #bb = psf.GradientBackground()
    #momo = bb.background_model(yfit, xfit, res[0])
    #ref_image[ref_extension].data = ref_image[ref_extension].data - momo

    ref_image_unmasked = np.copy(ref_image[ref_extension].data)
    if central_crop is not None:
        tmp_image = np.zeros(np.shape(ref_image[ref_extension].data))
        tmp_image[central_crop:-central_crop,central_crop:-central_crop] = ref_image[ref_extension].data[central_crop:-central_crop,central_crop:-central_crop]
        ref_image[ref_extension].data = tmp_image

        tmp_image2 = np.zeros(np.shape(noise_image))
        tmp_image2[central_crop:-central_crop,central_crop:-central_crop] = noise_image[central_crop:-central_crop,central_crop:-central_crop]
        noise_image = tmp_image2


    mask_extended = np.zeros((np.shape(ref_image[ref_extension].data)[0] + 2 * kernel_size,
                             np.shape(ref_image[ref_extension].data)[1] + 2 * kernel_size))
    mask_extended[kernel_size:-kernel_size, kernel_size:-kernel_size][ref_bright_mask_1] = 1.

    ref_extended = np.zeros((np.shape(ref_image[ref_extension].data)[0] + 2 * kernel_size,
                             np.shape(ref_image[ref_extension].data)[1] + 2 * kernel_size))
    ref_extended[kernel_size:-kernel_size, kernel_size:-
                 kernel_size] = np.array(ref_image[ref_extension].data, float)

    noise_extended = np.zeros((np.shape(ref_image[ref_extension].data)[0] + 2 * kernel_size,
                             np.shape(ref_image[ref_extension].data)[1] + 2 * kernel_size))
    noise_extended[kernel_size:-kernel_size, kernel_size:-
                 kernel_size] = np.array(noise_image, float)    

    #apply consistent mask
    ref_bright_mask = mask_extended > 0.
    mask_propagate = np.zeros(np.shape(ref_extended))
    mask_propagate[ref_bright_mask] = 1.
    #increase mask size to kernel size
    mask_propagate = convolve2d(mask_propagate, mask_kernel, mode='same')
    bright_mask = mask_propagate > 0.
    ref_extended[bright_mask] = 0.

    noise_extended[bright_mask] = 0.

    #replace saturated pixels with random noise or zero:
    bkg_sigma = np.std(ref_image_unmasked < np.median(ref_image_unmasked)) / (1.-2./np.pi)**0.5
    #apply consistent mask    
    ref_extended = cosmicray_lacosmic(ref_extended, sigclip=7, objlim = 7., satlevel = max_adu)[0]
    ref_image_unmasked = cosmicray_lacosmic(ref_image_unmasked, sigclip=7, objlim = 7, satlevel = max_adu)[0]
    ref_image_unmasked[bright_mask[kernel_size:-kernel_size,kernel_size:-kernel_size]] = 0. # np.random.randn(len(ref_image_unmasked[bright_mask[kernel_size:-kernel_size,kernel_size:-kernel_size]]))*bkg_sigma
 
    return np.array(ref_extended,dtype = float), bright_mask, np.array(ref_image_unmasked, dtype=float), np.array(noise_extended, dtype=float)
   
def open_images(setup, ref_image_directory, data_image_directory, ref_image_name,
                data_image_name, kernel_size, max_adu, ref_extension = 0, 
                data_image_extension = 0, log = None, subset = None):
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

    data_image = fits.open(os.path.join(data_image_directory_path, data_image_name), mmap=True)


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
