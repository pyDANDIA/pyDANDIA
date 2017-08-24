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

# Create or load the metadata file
reduction_metadata = stage0.create_or_load_the_reduction_metadata('../trials/', metadata_name='pyDANDIA_metadata.fits', verbose=True)

# Collect the image files
path_to_images = '../trials/data/'
images = glob.glob(path_to_images+'*fits')

# The configuration file specifies the header information for the input images
conf_dict = config.read_config('../Config/config.json')
gain =  conf_dict['gain']['value']
read_noise = conf_dict['ron']['value']

# Create new layer called 'data_inventory' in the metadata file (if it doesn't already exist)
reduction_metadata.create_a_new_layer(layer_name='data_inventory', data_structure=
                                      [
                                       ['IM_NAME','FWHM_X','FWHM_Y','SKY','CORR_XY'],
                                       ['S100','float','float','float','float'],
				       [None, 'arcsec', 'arcsec', 'ADU_counts', None]
				      ],
				      data_columns = None)

# For the set of given images, set the metadata information
for im in images:
    sky, fwhm_y, fwhm_x, corr_xy = starfind.starfind(im, plot_it=False, write_log=False)
    imname = im.split('/')[-1]
    # Add a new row to the data_inventory layer (if it doesn't already exist)
    reduction_metadata.add_row_to_layer(key_layer='data_inventory', new_row=[imname,fwhm_x,fwhm_y,sky,corr_xy])

# Save the updated layer to the metadata file
reduction_metadata.save_a_layer_to_file(metadata_directory='../trials/',metadata_name='pyDANDIA_metadata.fits',key_layer='data_inventory')
