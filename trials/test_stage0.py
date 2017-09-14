import sys
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import os
sys.path.append('/nethome/ebachelet/Desktop/pyDANDIA/pyDANDIA/')
from pyDANDIA import stage0
import time

os.system('rm pyDANDIA_metadata.fits')
data = fits.open('./data/lsc1m005-fl15-20170614-0130-e91.fits')

aa = stage0.read_the_config_file('../Config/')


reduction_metadata = stage0.create_or_load_the_reduction_metadata('./', metadata_name='pyDANDIA_metadata.fits', verbose=True)


stage0.update_reduction_metadata_with_config_file(reduction_metadata, aa)

data = stage0.find_all_images(reduction_metadata, './data/',verbose=True)
new_images=stage0.find_images_need_to_be_process(reduction_metadata, data, verbose=False)
if len(reduction_metadata.data_inventory[1])==0:
	stage0.create_reduction_metadata_data_inventory(reduction_metadata, new_images, status=0)
stage0.update_reduction_metadata_headers_summary_with_new_images(reduction_metadata, new_images)
open_image = stage0.open_an_image(reduction_metadata.data_architecture[1]['IMAGES_PATH'][0], new_images[0],image_index=0, verbose=True)

stage0.update_reduction_metadata_stamps(reduction_metadata, open_image,
                                     stamp_size=None, arcseconds_stamp_size=(60, 60),
                                     pixel_scale=None, number_of_overlaping_pixels=25,
                                     verbose=False)

stage0.set_bad_pixel_mask_directory(reduction_metadata, bpm_directory_path='./data/', verbose=False)



for new_image in new_images:
		open_image = stage0.open_an_image( reduction_metadata.data_architecture[1]['IMAGES_PATH'][0],new_image, image_index=0, verbose=True)
		bad_pixel_mask = stage0.open_an_image( reduction_metadata.data_architecture[1]['BPM_PATH'][0],new_image, image_index=2, verbose=True)
		

		stage0.construct_the_pixel_mask(open_image, bad_pixel_mask, [1,3],
                             saturation_level=65535, low_level=0)

stage0.construct_the_pixel_mask(open_image, bad_pixel_mask, [1,3],
                             saturation_level=65535, low_level=0)



stage0.update_reduction_metadata_data_inventory(reduction_metadata, new_images, status=1)
reduction_metadata.save_updated_metadata(reduction_metadata.data_architecture[1]['OUTPUT_DIRECTORY'][0],reduction_metadata.data_architecture[1]['METADATA_NAME'][0])


import pdb; pdb.set_trace()



