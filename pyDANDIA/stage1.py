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
reduction_metadata = stage0.create_or_load_the_reduction_metadata('./', metadata_name='pyDANDIA_metadata.fits', verbose=True)

path_to_images = '../trials/data/'
images = glob.glob(path_to_images+'*fits')

# The configuration file specifies the header information for the input images
conf_dict = config.read_config('../Config/config.json')
gain =  conf_dict['gain']['value']
read_noise = conf_dict['ron']['value']

# For the set of given images set the metadata information
for im in images:
    sky, fwhm_y, fwhm_x, corr_xy = starfind.starfind(im, plot_it=False, write_log=False)
    # Write to the metadata file
    reduction_metadata.create_data_inventory_layer([im])
    reduction_metadata.save_a_layer_to_file(metadata_directory='../trials/',metadata_name='pyDANDIA_metadata.fits',key_layer='data_inventory')
    reduction_metadata.add_column_to_layer(key_layer='data_inventory', new_column_name=['fwhm_x','fwhm_y','sky','corr_xy'], new_column_data=[fwhm_x,fwhm_y,sky,corr_xy], new_column_format=['float','float','float','float'], new_column_unit=['arcsec','arcsec','ADU counts',None])
    reduction_metadata.add_column_to_layer(key_layer='data_inventory', new_column_name='fwhm_y', new_column_data='float', new_column_format=None, new_column_unit='arcsec' )
    reduction_metadata.add_column_to_layer(key_layer='data_inventory', new_column_name='sky', new_column_data='float', new_column_format=None, new_column_unit='ADU counts' )
    reduction_metadata.add_column_to_layer(key_layer='data_inventory', new_column_name='corr_xy', new_column_data='float', new_column_format=None, new_column_unit=None )
