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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from pyDANDIA import psf
from pyDANDIA import image_handling
import scipy.ndimage as sn
from astropy.convolution import convolve
from astropy.convolution import Gaussian2DKernel


def background_fit(image1, master_mask = []):


    from pyDANDIA import psf
    y, x = np.indices(image1.shape)
    fit = psf.fit_background(image1, y, x, ~master_mask, background_model='Quadratic')
    background_model = psf.QuadraticBackground()
    background = background_model.background_model(y, x, fit[0])


    #background = np.median(image1[master_mask])
    return background

def background_mesh_perc(image1,perc=30,box_guess=300, master_mask = []):

    image = np.copy(image1)
    
    if (box_guess>image1.shape[0]/10) | (box_guess>image1.shape[1]/10):
    
        boxx = int(np.min(image1.shape[1])/10)
        boxy = int(np.min(image1.shape[0])/10)
        
       
    else:
        box = box_guess
        boxx = box
        boxy = box   
    xcen_range = (np.arange(0,image1.shape[1],boxx)+boxx/2).astype(int)
    ycen_range = (np.arange(0,image1.shape[0],boxy)+boxy/2).astype(int)

    
    halfboxx = int(boxx/2)
    halfboxy = int(boxy/2)
    
    percentile_bkg = np.zeros((len(ycen_range),len(xcen_range)))
    idx = 0
    jdx = 0

    result = np.zeros(image1.shape)
   
    for xcen in xcen_range:
        jdx = 0
        for ycen in ycen_range:
            try:

                sub_mask = master_mask[ycen - halfboxy:ycen + halfboxy+1,xcen - halfboxx:xcen+halfboxx+1]
                #positive = image[ycen - halfbox:ycen + halfbox+1,xcen - halfbox:xcen+halfbox+1] > perc5
                #val =  np.percentile(image[ycen - halfbox:ycen + halfbox+1,xcen - halfbox:xcen+halfbox+1][positive],perc)
                val = np.percentile(image[ycen - halfboxy:ycen + halfboxy+1,xcen - halfboxx:xcen+halfboxx+1][~sub_mask],perc)
                result[ycen - halfboxy:ycen + halfboxy+1,xcen - halfboxx:xcen+halfboxx+1][~sub_mask] = val
                percentile_bkg[jdx,idx] = val
            except:

                percentile_bkg[jdx, idx] = 0
            jdx += 1
        idx += 1
  
    #if boxx != boxy:
    #    import pdb; pdb.set_trace()

    #result = resize(percentile_bkg.T,(int(max(mask_shape_y)-min(mask_shape_y))+1,int(max(mask_shape_x)-min(mask_shape_x))+1) ,mode= 'symmetric')
    #result = resize(percentile_bkg,image1.shape ,mode= 'constant',order=0)

    #image[min(mask_shape_y):max(mask_shape_y)+1,min(mask_shape_x):max(mask_shape_x)+1] =result
    #result[zero_mask] =0.
    
   
    background = result
    #import pdb; pdb.set_trace()
    return background

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
    image_structure = image_handling.determine_image_struture(data_image_filename)
    data_image = fits.open(data_image_filename)
    ref_structure = image_handling.determine_image_struture(ref_image_filename)
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
    ref10pc = np.percentile(ref_image[ref_structure['sci']].data, 10)
    ref_image[ref_structure['sci']].data = ref_image[ref_structure['sci']].data - \
        np.percentile(ref_image[ref_structure['sci']].data, 0.1)

    # extend image size for convolution and kernel solution
    data_extended = np.zeros((np.shape(data_image[image_structure['sci']].data)))
    ref_extended = np.zeros((np.shape(ref_image[ref_structure['sci']].data)))
    data_extended = np.array(data_image[image_structure['sci']].data, float)
    ref_extended = np.array(ref_image[ref_structure['sci']].data, float)

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
                    max_adu, data_extension = 0, log = None, xshift = 0, yshift = 0, sigma_smooth = 0, central_crop = None,
                    subset = None, data_image1 = None, min_adu = None):
    '''
    reading difference image for constructing u matrix

    :param object string: reference imagefilename
    :param object string: reference image filename
    :param object integer: kernel size edge length of the kernel in px
    :param object float: index of the maximum adu values
    :return: images, mask
    '''


    #data_image1 is a large format image for processing subimages
    if data_image1 == None:
        data_image = fits.open(os.path.join(data_image_directory, data_image_name), mmap=True)
    if subset != None and data_image1 == None:
        #crop subimage
        data_image[data_extension].data=data_image[data_extension].data[subset[0]:subset[1],subset[2]:subset[3]]
    if subset != None and data_image1 != None:
        data_image = fits.HDUList(fits.PrimaryHDU(data_image1[data_extension].data[subset[0]:subset[1],subset[2]:subset[3]]))

    data_image = np.copy(data_image[data_extension].data)
    data_image_unmasked = np.copy(data_image)
    if sigma_smooth > 0:
        data_image = gaussian_filter(data_image, sigma=sigma_smooth)



    data_extended, data_image_unmasked = mask_the_image(data_image,max_adu,reference_mask,kernel_size)


    return data_extended, data_image_unmasked

def mask_the_reference(ref_image,reference_mask,kernel_size,max_adu):

    noise = np.copy(ref_image)
    ref_image,ref_mask = cosmicray_lacosmic(ref_image,sigclip=7, objlim = 7., satlevel = max_adu)

    bkg_image = background_mesh_perc(ref_image, master_mask = reference_mask)
    #bkg_image = np.percentile(data_image,10)
    ref_image = ref_image-bkg_image #- background_mesh_perc(data_image[data_extension].data,master_mask = reference_mask[kernel_size:-kernel_size,kernel_size:-kernel_size])
    #import pdb; pdb.set_trace()
    ref_image[reference_mask] = 0
   
    ref_image_unmasked = np.copy(ref_image)


    
    ref_extended = np.zeros((np.shape(ref_image)[0] + 2 * kernel_size, np.shape(ref_image)[1] + 2 * kernel_size))
    mask_extended = np.ones((np.shape(ref_image)[0] + 2 * kernel_size, np.shape(ref_image)[1] + 2 * kernel_size)).astype(bool)
    noise_extended = np.ones((np.shape(ref_image)[0] + 2 * kernel_size, np.shape(ref_image)[1] + 2 * kernel_size))
    
    ref_extended[kernel_size:-kernel_size, kernel_size:-kernel_size] = np.array(ref_image, float)
    mask_extended[kernel_size:-kernel_size, kernel_size:-kernel_size] = reference_mask
    noise_extended[kernel_size:-kernel_size, kernel_size:-
                 kernel_size] = np.array(noise, float)

    #mask_extended = sn.morphology.binary_dilation(mask_extended, iterations=1*kernel_size)
    ref_extended[mask_extended] = 0
    noise_extended[mask_extended] = 1

   

    return ref_extended, ref_image_unmasked,mask_extended, bkg_image,noise_extended


def mask_the_image(data_image,max_adu,reference_mask,kernel_size):
    #import pdb; pdb.set_trace()

    data_image,data_mask = cosmicray_lacosmic(data_image,sigclip=7, objlim = 7., satlevel = max_adu)

    #bkg_image = background_fit(data_image, master_mask = reference_mask[kernel_size:-kernel_size,kernel_size:-kernel_size])
    bkg_image = background_mesh_perc(data_image, master_mask = reference_mask[kernel_size:-kernel_size,kernel_size:-kernel_size])
    #bkg_image = np.percentile(data_image,10)
    data_image = data_image-bkg_image #- background_mesh_perc(data_image[data_extension].data,master_mask = reference_mask[kernel_size:-kernel_size,kernel_size:-kernel_size])
    #import pdb; pdb.set_trace()
    data_image[reference_mask[kernel_size:-kernel_size, kernel_size:-kernel_size]] = 0
   
    mask_extended = reference_mask
    #mask_extended = sn.morphology.binary_dilation(mask_extended, iterations=1*kernel_size)
    
    data_extended = np.zeros((np.shape(data_image)[0] + 2 * kernel_size, np.shape(data_image)[1] + 2 * kernel_size))
    data_extended[kernel_size:-kernel_size, kernel_size:-kernel_size] = np.array(data_image, float)
    data_extended[mask_extended] = 0.
    

    data_image_unmasked = np.copy(data_image)

    # extend image size for convolution and kernel solution
   

    return data_extended, data_image_unmasked, bkg_image





def open_reference(setup, ref_image_directory, ref_image_name, kernel_size, max_adu, ref_extension = 0, log = None, central_crop = None, subset = None, ref_image1 = None, min_adu = None, master_mask = [], external_weight = None):
    '''
    reading difference image for constructing u matrix

    :param object string: reference imagefilename
    :param object string: reference image filename
    :param object integer: kernel size edge length of the kernel in px
    :param object float: index of the maximum adu values
    :return: images, mask
    '''

    #import pdb; pdb.set_trace()

    if ref_image1 == None:
        ref_image = fits.open(os.path.join(ref_image_directory, ref_image_name), mmap=True)
    if subset != None and ref_image1 == None:
        ref_image=ref_image[ref_extension].data[subset[0]:subset[1],subset[2]:subset[3]]
        master_mask=master_mask[subset[0]:subset[1],subset[2]:subset[3]]
    if subset != None and ref_image1 != None:
        ref_image = fits.HDUList(fits.PrimaryHDU(ref_image1[ref_extension].data[subset[0]:subset[1],subset[2]:subset[3]]))
        master_mask=master_mask[subset[0]:subset[1],subset[2]:subset[3]]

    ref_image = np.copy(ref_image)
    ref_image,ref_mask  = cosmicray_lacosmic(ref_image,sigclip=7, objlim = 7., satlevel = max_adu)



    #bkg_image = background_mesh_perc(ref_image,master_mask = master_mask)
    #bkg_image = np.median(ref_image[~master_mask])
   # bkg_image = background_fit(ref_image, master_mask=master_mask)
    bkg_image = background_mesh_perc(ref_image, master_mask=master_mask)
    #bkg_image = np.percentile(ref_image,10)
    if external_weight is not None:
        try:
            noise_image = external_weight + np.copy(ref_image)
        except:
            noise_image = np.zeros(np.shape(ref_image))
            print('format mismatch (noise model construction)')
    else:
        noise_image = np.copy(ref_image)


    #noise_image = gaussian_filter(noise_image, sigma=kernel_size/2)
    ref_image = ref_image - bkg_image
    ref_image_unmasked = np.copy(ref_image)






    mask_extended = np.ones((np.shape(ref_image)[0] + 2 * kernel_size,
                             np.shape(ref_image)[1] + 2 * kernel_size)).astype(bool)
    mask_extended[kernel_size:-kernel_size, kernel_size:-kernel_size] = master_mask
    
    #mask_extended = sn.morphology.binary_dilation(mask_extended, iterations=5*kernel_size)
    
    ref_extended = np.zeros((np.shape(ref_image)[0] + 2 * kernel_size,
                             np.shape(ref_image)[1] + 2 * kernel_size))
    ref_extended[kernel_size:-kernel_size, kernel_size:-
                 kernel_size] = np.array(ref_image, float)

    noise_extended = np.zeros((np.shape(ref_image)[0] + 2 * kernel_size,
                             np.shape(ref_image)[1] + 2 * kernel_size))
    noise_extended[kernel_size:-kernel_size, kernel_size:-
                 kernel_size] = np.array(noise_image, float)
                 
    #mask_extended = sn.morphology.binary_dilation(mask_extended, iterations=20)

    ref_extended[mask_extended] = 0.

    #noise_extended = np.ones(ref_extended.shape)
    noise_extended[mask_extended] = 0.


    ref_image_unmasked[master_mask] = 0. # np.random.randn(len(ref_image_unmasked[bright_mask[kernel_size:-kernel_size,kernel_size:-kernel_size]]))*bkg_sigma
    return np.array(ref_extended,dtype = float), mask_extended, np.array(ref_image_unmasked, dtype=float), np.array(noise_extended, dtype=float)





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
