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


def find_images_to_process(reduction_metadata, images_directory_path= None, verbose=False):



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

def update_the_metadata_with_new_images(reduction_metadata, images, verbose=False):

	try :
		layer = reduction_metadata.reduction_status

		if layer == [None, None]:	

			new_images = images
			reduction_metadata.create_reduction_status_layer(new_images)
			reduction_metadata.create_headers_summary_layer(new_images)
			reduction_metadata.create_data_inventory_layer(new_images)
			new_images = None

		else:

			new_images = []
	
			old_images = layer[1]['IMAGES']

			for name in images :

				if name not in old_images:
				
					new_images.append(name)

	except:

		print('Something went wrong on images/metadata matching !')


	if new_images:

				### to work
				layer_reduction = reduction_metadata.reduction_status
				new_row_reduction = [[None]]*len(layer_reduction.keys())
				layer_headers = reduction_metadata.headers_summary
				new_row_header = [[None]]*len(layer_reduction.keys())
				layer_data = reduction_metadata.data_inventory
				new_row_data = [[None]]*len(layer_reduction.keys())

				for image_name in new_images:

					new_row_reduction[0] = image_name
					new_row_header[0] = image_name
					new_row_data[0] = image_name

					reduction_metadata.add_row(reduction_status, new_row_reduction)
					reduction_metadata.add_row(reduction_status, new_row_header)
					reduction_metadata.add_row(reduction_status, new_row_data)
				
	else:
			
		pass
def find_images_already_process(reduction_metadata, verbose=False):


	try:

		images_already_treated = reduction_metadata['REDUCTION_STATUS']['NAMES']

		if verbose == True:

			print('Find '+str(len(images_already_treated))+' images already process.')

		return images_already_treated

	except:
		
		if verbose == True:

			print('Could not find any images already process!')

		return []

def remove_images_already_process(list_of_images, list_of_already_process_image, verbose=False):

	try:

		if len(list_of_already_process_image) != 0:

			for image in list_of_images:

				if image in list_of_already_process_images_list :

					list_of_images.remove(image)

	except:
		if verbose == True:

			print('Something went wrong on removing images already process !')

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

def add_table_to_the_metadata(table_name, table_data, table_columns_names, table_format, open_metadata, 
			      output_metadata_directory, metadata_name='pyDANDIA_metadata.fits', verbose=False):

	columns = []

	for i in xrange(len(table_data[0])):

		column = fits.Column(name = table_columns_names[i], format = table_format[i], array=table_data[:,i])
		columns.append(column)

	tbhdu = fits.BinTableHDU.from_columns(columns)
	tbhdu.name = table_name
	
	if table_name in [i.name for i in open_metadata]:
		
		open_metadata[table_name] = tbhdu
	else:
	
		open_metadata.append(tbhdu)



def add_column_to_a_table_in_the_metadata(table_name, table_data, table_columns_names, table_format, open_metadata, 
			      output_metadata_directory, metadata_name='pyDANDIA_metadata.fits', verbose=False):

	columns = []

	for i in xrange(len(table_data[0])):

		column = fits.Column(name = table_columns_names[i], format = table_format[i], array=table_data[:,i])
		columns.append(column)

	tbhdu = fits.BinTableHDU.from_columns(columns)
	tbhdu.name = table_name
	
	if table_name in [i.name for i in open_metadata]:
		
		open_metadata[table_name] = tbhdu
	else:
	
		open_metadata.append(tbhdu)



def add_image_to_the_metadata(image_name, image_data, open_metadata, 
			      output_metadata_directory, metadata_name='pyDANDIA_metadata.fits', verbose=False):

	new_hdu = fits.ImageHDU(image_data, name = image_name)

	if image_name in [i.name for i in open_metadata]:
		
		open_metadata[image_name] = new_hdu
	else:
	
		open_metadata.append(new_hdu)

def save_the_metadata(open_metadata, output_metadata_directory, metadata_name='pyDANDIA_metadata.fits', verbose=False):

	open_metadata.writeto(output_metadata_directory+metadata_name, overwrite=True)


