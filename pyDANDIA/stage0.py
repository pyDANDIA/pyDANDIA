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

PIPELINE_CONFIGURATION = config.read_config('../Config/config.json')




def create_or_load_the_reduction_metadata(output_metadata_directory, metadata_name='pyDANDIA_metadata.fits', verbose=False):

	try:

		meta_data_exist = [i for i in os.listdir(output_metadata_directory) if (i == metadata_name)]	
		
		if meta_data_exist == [] :

			
			reduction_metadata = fits.HDUList()

			if verbose == True:

				print('Successfully create the reduction metadata')

		else :

			reduction_metadata = fits.open(output_metadata_directory + metadata_name)
			if verbose == True:

				print('Successfully found the reduction metadata')
	except:

		if verbose == True:

			print('No metadata created or loaded : check this!')

		sys.exit(1)

	
	return reduction_metadata


def find_images_to_process(images_directory_path, images_name_definition='.fits', verbose=False):


	try:

		list_of_images = [i for i in os.listdir(images_directory_path) if (i[-5:] == images_name_definition) ]	
		
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

def find_images_already_process(reduction_metadata, verbose=False):


	try:

		images_already_treated = reduction_metadata['images']['names']

		if verbose == True:

			print('Find '+str(len(images_already_treated))+' images to treat.')

		return images_already_treated

	except:
		
		if verbose == True:

			print('Could not find any images already treated!')

		return []

def remove_images_already_process(list_of_images, list_of_already_process_image, verbose=False):

	try:

		if len(list_of_already_process_image) != 0:

			for image in list_of_images:

				if image in list_of_already_process_images_list :

					list_of_images.remove(image)

	except:
		if verbose == True:

			print('Something went wrong on removing images already treated !')

		pass

	if verbose == True:

		print('The total number of frames to treat is :'+str(len(list_of_images)))

	return list_of_images



def open_an_image(image_directory_path, image_name, verbose=False):
	
	try:

		image_data = fits.open(image_directory_path+image_name)
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

	except:
		
		bad_pixel_mask = np.zeros(open_image[0].data.shape, int)



	return bad_pixel_mask
	

def construct_the_variables_star_mask():
	pass

def construct_the_saturated_pixel_mask(open_image, saturation_level = 65535):
	
	try:
		data = open_image[0].data
		
		mask = data >= saturation_level

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
	
def construct_the_stamps(open_image, arcseconds_stamp_size=60, number_of_overlaping_pixels=25,
				 number_of_image_division=None, verbose=False):

	image = open_image[0].data

	full_image_y_size, full_image_x_size = image.shape

	if number_of_image_division:	

		pass	
	
	else:
		try :

			header = open_image[0].header
			pixel_scale = float(header['PIXSCALE'])
	
		except:

			try:

				pixel_scale = PIPELINE_CONFIGURATION['pixel_scale'] 
		
			except:
				print('No pixel scale found!')
				sys.exit(1)

		
		
	y_stamp_size = int(arcseconds_stamp_size/pixel_scale)
	x_stamp_size = int(arcseconds_stamp_size/pixel_scale)

	y_stamps_center = np.arange(x_stamp_size/2,full_image_x_size, x_stamp_size)
	x_stamps_center = np.arange(y_stamp_size/2,full_image_y_size, y_stamp_size)


	stamps_center_y, stamps_center_x = np.meshgrid(y_stamps_center, x_stamps_center)

	stamps_y_min = stamps_center_y - y_stamp_size/2-number_of_overlaping_pixels
	stamps_y_max = stamps_center_y + y_stamp_size/2+number_of_overlaping_pixels

	stamps_x_min = stamps_center_x - x_stamp_size/2-number_of_overlaping_pixels
	stamps_x_max = stamps_center_x + x_stamp_size/2+number_of_overlaping_pixels

	stamps_coordinates=np.zeros(stamps_x_min.shape)
	stamps_list = []
	count = 0
	for i in xrange(len(stamps_coordinates)):

			for j in xrange(len(stamps_coordinates[0])):
				stamps_list.append([count,max(stamps_y_min[i,j],0),min(stamps_y_max[i,j],full_image_y_size),
							  max(stamps_x_min[i,j],0),min(stamps_x_max[i,j],full_image_x_size)])
				count += 1

	return np.array(stamps_list)



