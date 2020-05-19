from astropy.io import fits

import matplotlib.pyplot as plt

from skimage import data
from skimage.feature import register_translation
from skimage.feature.register_translation import _upsampled_dft
from scipy.ndimage import fourier_shift

import numpy as np
from photutils import background, detection, DAOStarFinder
from astropy.stats import sigma_clipped_stats
from sys import exit
import os

from skimage import transform as tf
from pyDANDIA import stage4

def modify_header(combined_image_file, list_of_images_used, 
                  read_noise_kw='RDNOISE',
                  exptime_kw='EXPTIME',
                  saturation_kw='SATURATE',
                  linearity_kw='MAXLIN'):
    """
    Modify the header of a combined reference image file
    :param str combined_image_file: fill path to the image file (fits)
    :param list list_of_images_used: a list of the images that were combined (full paths)
                                     including the image all others were aligned to.
    :params str read_noise_kw, exptime_kw, saturation_kw, linearity_kw:
                                     corresponding header keywords
    """
    # Get header structure from the first image
    header = fits.getheader(list_of_images_used[0])
    image_list = [i.split('/')[-1] for i in list_of_images_used]
    read_noise_terms = np.array([fits.getheader(i)[read_noise_kw] for i in list_of_images_used])
    new_read_noise = np.sqrt(np.sum(read_noise_terms**2))
    exptime_terms = np.array([fits.getheader(i)[exptime_kw] for i in list_of_images_used])
    new_exptime = np.sum(exptime_terms)
    if saturation_kw !='':
        saturation_terms = np.array([fits.getheader(i)[saturation_kw] for i in list_of_images_used])
        new_saturate = np.sum(saturation_terms)
    if linearity_kw !='':
        linearity_terms = np.array([fits.getheader(i)[linearity_kw] for i in list_of_images_used])
        new_maxlin = np.sum(linearity_terms)
    
    # Add new header keywords for the images used in the construction of the
    # combined reference
    count = 1
    for image in image_list:
        header.set('REFIMG'+str(count), image)
        count = count +1
    
    # Modify remaining header keywords
    header.set('RDNOISE', new_read_noise)
    header.set('EXPTIME', new_exptime)
    if saturation_kw !='':
        header.set('SATURATE', new_saturate)
    if linearity_kw !='':
        header.set('MAXLIN', new_maxlin)
    # Update the image file with the new header data
    data = fits.getdata(combined_image_file)
    fits.update(combined_image_file, data, header)
    print('Modified header information for image: %s' % combined_image_file.split('/')[-1])
    
def extract_sources(image_file):
    """
    Find the sources in an image
    :param str image: full path to the image file (fits)
    
    :return: sources identified, fwhm 
    """
    
    hdul = fits.open(image_file)
    
    try:
        fwhm = hdul[0].header['L1FWHM']
    except:
        fwhm = 3.0
    
    image_data = hdul[0].data
    image = np.copy(image_data)
    central_region_x, central_region_y = np.shape(image)
    center_x, center_y = int(central_region_x / 2), int(central_region_y / 2)
    
    # Evaluate mean, median and standard deviation for the image
    mean, median, std = sigma_clipped_stats(image, sigma=3.0, maxiters=5)
    daofind = DAOStarFinder(fwhm=3.0, threshold=3.*std, exclude_border=True)
    sources = daofind.find_stars(image - median)
    
    return sources, fwhm


def find_xy_shifts(reference_image, target_image, edgefraction=0.5):
    """
    Find the pixel offsets between two images
    
    :param object reference_image: the reference image data (i.e image.data)
    :param object target_image: the image data of interest (i.e image.data)
    :param float edgefraction: the percentage of images use for the shift computation (smaller = faster, [0,1])
    
    :return: [x_new_center, y_new_center, x_shift, y_shift], the new center and the correspondind shift of the target image
    :rtype: array_like
    """
    
    reference_shape = reference_image.shape
    
    if reference_shape != target_image.shape:
        print('The reference image and the target image dimensions does not match! Aborted.')
        exit()
    
    x_center = int(reference_shape[0] / 2)
    y_center = int(reference_shape[1] / 2)
    
    half_x = int(edgefraction * float(reference_shape[0]) / 2)
    half_y = int(edgefraction * float(reference_shape[1]) / 2)
    
    reduce_image = target_image
    reduce_template = reference_image
    
    from skimage.feature import register_translation
    shifts, errors, phasediff = register_translation(reduce_template, reduce_image, 10)
    
    x_shift = shifts[1]
    y_shift = shifts[0]
    
    x_new_center = -x_shift + x_center
    y_new_center = -y_shift + y_center
    
    return x_new_center, y_new_center, x_shift, y_shift


def combine_images(list_of_images):

	#weighted mean so far
	weights =  np.array([ np.abs(i)**0.5 for i in list_of_images])
	weights = 1
	stacked = np.sum(list_of_images*weights,axis=0)/np.sum(weights,axis=0)
	
	return stacked		

def resample_image(new_images, reference_image):

	from skimage.measure import ransac

	reference_image_hdu = fits.open(reference_image, memmap=True)
	reference_image_data = np.copy(reference_image_hdu[0].data)
	ref_sources, ref_fwhm = extract_sources(reference_image)

	print("Starting image resampling...")

	shifteds = []
	for new_image in new_images:
		print('Resampling image '+new_image)

		new_image_hdu = fits.open(new_image, memmap=True)
		new_image_data = np.copy(new_image_hdu[0].data)
		new_image_sources, new_image_fwhm = extract_sources(new_image)        
		x_new_center, y_new_center, x_shift, y_shift = find_xy_shifts(reference_image_data.astype(float), new_image_data.astype(float))


		shifted = np.copy(new_image_data)
		iteration = 0


		while iteration < 1:
			data_sources = np.copy(new_image_sources)

	
			try:
				if iteration > 0:

					x_shift = 0
					y_shift = 0
					original_matrix = model_final.params

				else:
					original_matrix = np.identity(3)

				pts_data, pts_reference, e_pos = stage4.crossmatch_catalogs(ref_sources, new_image_sources, -x_shift, -y_shift)

				pts_reference2 = np.copy(pts_reference)

				model_robust, inliers = ransac((pts_reference2[:5000, :2] , pts_data[:5000, :2] ), tf.AffineTransform,
							   min_samples=min(50, int(0.1 * len(pts_data[:5000]))),
							   residual_threshold=0.05, max_trials=1000)

				if len(pts_data[:5000][inliers])<10:
					raise ValueError("Not enough matching stars! Switching to translation")
				model_final = np.dot(original_matrix, model_robust.params)


			except:

				model_final = tf.SimilarityTransform(translation=(-x_shift, -y_shift)).params
			shifted = tf.warp(new_image_data, inverse_map=model_final, output_shape=new_image_data.shape, order=5,mode='constant', cval=0, clip=True, preserve_range=True)

	

			iteration += 1
		shifteds.append(shifted)
	all_images = shifteds+[reference_image_data]
	return all_images

def test_combine_images(image_path='/work/Ytsapras/ROME-REA/Images/ip/', 
                        image_list=['coj1m003-fa19-20200410-0033-e91.fits',
                                    'coj1m003-fa19-20200410-0024-e91.fits',
                                    'coj1m003-fa19-20200410-0027-e91.fits',
                                    'coj1m003-fa19-20200410-0030-e91.fits',
                                    'coj1m003-fa19-20200410-0036-e91.fits']):
    """
    Test combine images and print diagnostic stats
    
    :param str image_path: The full path to the image directory
    :param list(str) image_list: the list of images. !IMPORTANT! The first image 
                                 must always be the reference that all others
                                 will be aligned to.
    """
    import copy
    # Read in the images 
    ref_image = image_path+image_list[0]
    
    images = [image_path+img for img in image_list[1:]]
    
    # Create a deep copy of the list of images
    list_of_images_used = copy.deepcopy(images)
    # Add the ref_image first in the list
    list_of_images_used.insert(0,ref_image)
    
    aligned_images = resample_image(images, ref_image)
    
    #stack with all images
    stacked5 = combine_images(aligned_images)
    hdu = fits.PrimaryHDU(stacked5)
    
    hdul = fits.HDUList([hdu])
    hdul.writeto(image_path+'stack5.fits',overwrite=True)
    modify_header(image_path+'stack5.fits', list_of_images_used)
    
    #stack 3 best seeing
    stacked3 = combine_images(aligned_images[2:])
    hdu = fits.PrimaryHDU(stacked3)
    
    hdul = fits.HDUList([hdu])
    hdul.writeto(image_path+'stack3.fits',overwrite=True)
    modify_header(image_path+'stack3.fits', list_of_images_used)
    
    ### Compare number of sources
    mean0, median0, std0 = sigma_clipped_stats(aligned_images[-1], sigma=3.0, maxiters=5)
    daofind = DAOStarFinder(fwhm=3.0, threshold=3.*std0, exclude_border=True)
    sources0 = daofind.find_stars(aligned_images[-1] - median0)
    
    mean5, median5, std5 = sigma_clipped_stats(stacked5, sigma=3.0, maxiters=5)
    daofind = DAOStarFinder(fwhm=3.0, threshold=3.*std5, exclude_border=True)
    sources5 = daofind.find_stars(stacked5 - median5)
    
    mean3, median3, std3 = sigma_clipped_stats(stacked3, sigma=3.0, maxiters=5)
    daofind = DAOStarFinder(fwhm=3.0, threshold=3.*std3, exclude_border=True)
    sources3 = daofind.find_stars(stacked3 - median3)
    
    print('ref:',len(sources0),mean0, median0, std0)
    print('stack5:',len(sources5),mean5, median5, std5)
    print('stack3:',len(sources3),mean3, median3, std3)

