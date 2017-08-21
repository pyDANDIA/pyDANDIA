######################################################################
#
# stage1.py - For a set of images, provide in a metadata fits file: 
#                 FWHM (in pix)
#		  sky background
#		  gain
#		  dark current
#		  readout noise
#
# dependencies:
#       starfind.py
#
# Developed by the Yiannis Tsapras
# as part of the ROME/REA LCO Key Project.
#
# version 0.1a (development)
#
# Last update: 27 Jul 2017
######################################################################

import config
import starfind
import os
import sys
import glob

sys.path.append('../pyDANDIA/')
import metadata
import stage0
reduction_metadata = stage0.create_or_load_the_reduction_metadata('.', metadata_name='test_metadata.fits', verbose=True)

path_to_images = '../trials/data/'
images = glob.glob(path_to_images+'*fits')

# The configuration file specifies the header information for the input images
conf_dict = config.read_config('../Config/config.json')
gain =  conf_dict['gain']['value']
read_noise = conf_dict['ron']['value']

# Set up holiding arrays
names_arr = []
sky_arr = []
fwhm_x_arr = []
fwhm_y_arr = []
corr_xy_arr = []

# For the set of given images set the metadata information
for im in images:
    sky, fwhm_y, fwhm_x, corr_xy = starfind.starfind(im, plot_it=False, write_log=False)
    names_arr.append(im.split('/')[-1])
    sky_arr.append(sky)
    fwhm_x_arr.append(fwhm_y)
    fwhm_y_arr.append(fwhm_x)
    corr_xy_arr.append(corr_xy)

# Write to the metadata file
reduction_metadata.add_column_to_layer(key_layer='data_inventory', new_column_name='im_name', new_column_data=names_arr, new_column_format='str', new_column_unit=None )
reduction_metadata.add_column_to_layer(key_layer='data_inventory', new_column_name='fwhm_x', new_column_data=fwhm_x, new_column_format='float', new_column_unit='arcsec' )
reduction_metadata.add_column_to_layer(key_layer='data_inventory', new_column_name='fwhm_y', new_column_data=fwhm_y, new_column_format='float', new_column_unit='arcsec' )
reduction_metadata.add_column_to_layer(key_layer='data_inventory', new_column_name='sky', new_column_data=sky, new_column_format='float', new_column_unit='ADU counts' )
reduction_metadata.add_column_to_layer(key_layer='data_inventory', new_column_name='corr_xy', new_column_data=corr_xy, new_column_format='float', new_column_unit=None )
reduction_metadata.save_a_layer_to_file(metadata_directory='.',metadata_name='test_metadata.fits',key_layer='data_inventory' )
