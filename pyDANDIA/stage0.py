######################################################################
#                                                                   
# stage0.py - First stage of the pipeline. Handle data, create bad pixels mask...
# More details in individual fonctions.

#
# dependencies:
#      numpy 1.8+
#      astropy 1.0+ 
######################################################################

import numpy as np
import os
from astropy.io import fits
import sys


import config
from astropy.table import Table
from astropy.nddata import Cutout2D

import metadata


def open_the_variables_catalog(variables_catalog_directory, variables_catalog_name):


	variable_catalog = None
	pass

def read_the_config_file(config_directory, config_file_name = 'config.json'):

	


	pipeline_configuration = config.read_config(config_directory+config_file_name)
	
	return pipeline_configuration

def create_or_load_the_reduction_metadata(output_metadata_directory, metadata_name='pyDANDIA_metadata.fits', verbose=False):

	try:

		meta_data_exist = [i for i in os.listdir(output_metadata_directory) if (i == metadata_name)]	

		if meta_data_exist == [] :

			
				reduction_metadata = metadata.MetaData()

				reduction_metadata.create_metadata_file(output_metadata_directory, metadata_name)				


				if verbose == True:

					print('Successfully create the reduction metadata')

		else :

			reduction_metadata = metadata.MetaData()

			reduction_metadata.load_all_metadata(output_metadata_directory, metadata_name)
			if verbose == True:

				print('Successfully found the reduction metadata')
	except:

		if verbose == True:

			print('No metadata created or loaded : check this!')

		sys.exit(1)

	
	return reduction_metadata


def find_all_images(reduction_metadata, images_directory_path= None, verbose=False):



	try:

		path = reduction_metadata.data_architecture[1]['IMAGES_PATH']
		
	except:
		
		if images_directory_path :
			
			path =  images_directory_path

			reduction_metadata.add_column_to_layer('data_architecture', 'images_path', [path])			
		

	try:

		list_of_images = [i for i in os.listdir(path)]	
		
		if list_of_images == [] :
		
			if verbose == True:

				print('No images to process. I take a rest :)')

			return None
			

		else :

			if verbose == True:

				print('Find '+str(len(list_of_images))+' images to treat.')

			return list_of_images

	except:

		if verbose == True:

				print('Something went wrong on images search!')
	

		return None

def find_images_need_to_be_process(reduction_metadata, images, verbose=False):

	try :
		layer = reduction_metadata.reduction_status

		if layer == [None, None]:	

			new_images = images
			
		else:

			new_images = []
	
			old_images = layer[1]['IMAGES']

			for name in images :

				if name not in old_images:
				
					new_images.append(name)

	except:

		print('Something went wrong on images/metadata matching !')


	return new_images



def open_an_image(image_directory_path, image_name, verbose=False):

	try:

		image_data = fits.open(image_directory_path+image_name, mmap=True)
		if verbose == True:

			print(image_name+' open : OK')
	
		return image_data
	except:
		if verbose == True:

			print(image_name+' open : not OK!')

		return None


def construct_the_bad_pixel_mask(open_image, image_bad_pixel_mask_layer = 2):

	try:

		bad_pixel_mask = open_image[image_bad_pixel_mask_layer].data

		#BANZAI definition of bp == 1 or bp == 3		
		index_saturation = np.where((bad_pixel_mask==2))
		bad_pixel_mask[index_saturation] = 0

		index_saturation = np.where((bad_pixel_mask==3))
		bad_pixel_mask[index_saturation] = 1
 
	except:
		
		bad_pixel_mask = np.zeros(open_image[0].data.shape, int)



	return bad_pixel_mask
	

def construct_the_variables_star_mask(open_image, variable_star_pixels = 10):

	try:

		RA_range = [265, 285]
		DEC_range = [-35,-25]


		data = open_image[0].data
		
		if saturation_level:

			pass
	
	except:
		
		saturated_pixel_mask = np.zeros(open_image[0].data.shape,int)

	pass

def construct_the_saturated_pixel_mask(open_image, saturation_level = None):
	
	try:
		data = open_image[0].data
		
		if saturation_level:

			pass
		else:

			saturation_level = open_image[0].header['SATURATE']
		
		mask = 	open_image[0].data >= saturation_level
		saturated_pixel_mask = mask.astype(int)

	except:
		
		saturated_pixel_mask = np.zeros(open_image[0].data.shape,int)


	return saturated_pixel_mask
	
def construct_the_low_level_pixel_mask(open_image, low_level = 0):
	
	try:
		data = open_image[0].data
		
		mask = data <= low_level

		low_level_pixel_mask = mask.astype(int)

	except:
		
		low_level_pixel_mask = np.zeros(open_image[0].data.shape,int)


	return low_level_pixel_mask

def construct_the_pixel_mask(open_image):

	try:
		bad_pixel_mask = construct_the_bad_pixel_mask(open_image)
	
		#variables_pixel_mask = construct_the_variables_star_mask()
	
		saturated_pixel_mask = construct_the_saturated_pixel_mask(open_image)
	
		low_level_pixel_mask = construct_the_low_level_pixel_mask(open_image)
	
		master_mask = np.zeros(open_image[0].data.shape,int)

		list_of_masks = [bad_pixel_mask, saturated_pixel_mask,low_level_pixel_mask]

		for index,mask in enumerate(list_of_masks[::-1]):

			master_mask += mask*2**index 
		
		return master_mask

	except:

		master_mask = np.zeros(open_image[0].data.shape,int)

	return master_mask	

def save_the_pixel_mask_in_image(open_image, master_mask, image_name, reduction_metadata):

	bad_pixels_mask = fits.ImageHDU(master_mask)
	bad_pixels_mask.name = 'MASTER_PIXEL_MASK'
	
	try:
		open_image['MASTER_PIXEL_MASK'] = bad_pixels_mask
	except:

		open_image.append(bad_pixels_mask)

	image_directory = reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]
		
	open_image.writeto(image_directory+image_name, overwrite=True)
def construct_the_stamps(open_image, stamp_size = None, arcseconds_stamp_size=(60,60), pixel_scale = None, 
			 number_of_overlaping_pixels=25, verbose=False):

	image = open_image[0].data

	full_image_y_size, full_image_x_size = image.shape

	if stamp_size:	

		y_stamp_size = stamp_size[0]
		x_stamp_size = stamp_size[1]
	
	else:
		try:

			y_stamp_size = int(arcseconds_stamp_size[0]/pixel_scale)
			x_stamp_size = int(arcseconds_stamp_size[1]/pixel_scale)
		
		except:
			print('No pixel scale found!')
			sys.exit(1)

		
		
	

	x_stamps_center = np.arange(x_stamp_size/2,full_image_x_size, x_stamp_size)
	y_stamps_center = np.arange(y_stamp_size/2,full_image_y_size, y_stamp_size)


	stamps_center_x, stamps_center_y = np.meshgrid(y_stamps_center, x_stamps_center)

	stamps_y_min = stamps_center_y - y_stamp_size/2-number_of_overlaping_pixels
	mask = 	stamps_y_min < 0
	stamps_y_min[mask] = 0

	stamps_y_max = stamps_center_y + y_stamp_size/2+number_of_overlaping_pixels
	mask = 	stamps_y_max > full_image_y_size
	stamps_y_min[mask] = full_image_y_size

	stamps_x_min = stamps_center_x - x_stamp_size/2-number_of_overlaping_pixels
	mask = 	stamps_x_min < 0
	stamps_x_min[mask] = 0

	stamps_x_max = stamps_center_x + x_stamp_size/2+number_of_overlaping_pixels
	mask = 	stamps_x_max > full_image_x_size
	stamps_x_min[mask] = full_image_x_size

	stamps = [[j*(i+1),stamps_y_min[i,j],stamps_y_max[i,j],stamps_x_min[i,j],stamps_x_max[i,j]] 
		  for i in range(stamps_x_min.shape[0]) for j in range(stamps_x_min.shape[1])]


	return np.array(stamps)



