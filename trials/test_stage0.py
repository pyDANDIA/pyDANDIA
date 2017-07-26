import sys
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits
import os
sys.path.append('../pyDANDIA/')
import stage0

os.system('rm pyDANDIA_metadata.fits')
data = fits.open('./data/lsc1m005-fl15-20170614-0130-e91.fits')

aa = stage0.read_the_config_file('../Config/')
aa['pixel_scale'] = 0.389
stamps =  stage0.construct_the_stamps(data,arcseconds_stamp_size=(60,60),pixel_scale=aa['pixel_scale'])


reduction_metadata = stage0.create_or_load_the_reduction_metadata('./', metadata_name='pyDANDIA_metadata.fits', verbose=True)

data = stage0.find_all_images(reduction_metadata, './data/',verbose=True)

new_images = stage0.find_images_need_to_be_process(reduction_metadata, data,verbose=True)
aa = stage0.read_the_config_file('../Config/')


for data in new_images:
	import pdb; pdb.set_trace()
	open_image = stage0.open_an_image( reduction_metadata.data_architecture[1]['IMAGES_PATH'][0],data, verbose=True)
	

			stamps = stage0.construct_the_stamps(open_image, pixel_scale = 0.389)
			data_structure = [['ID','Y_MIN','Y_MAX','X_MIN','X_MAX'],
			 		 ]
			reduction_metadata.create_a_new_layer('stamps', data_structure, data_columns = stamps)
			reduction_metadata.save_a_layer_to_file( './', 'pyDANDIA_metadata.fits', 'stamps')
	master_mask = stage0.construct_the_pixel_mask(open_image)
	stage0.save_the_pixel_mask_in_image(open_image, master_mask, data, reduction_metadata)
import pdb; pdb.set_trace()


stage0.add_table_to_the_metadata('IMAGES', images, ['NAMES','ID'], ['100A','I'],reduction_metadata, 
			      './', metadata_name='pyDANDIA_metadata.fits')
stage0.add_table_to_the_metadata('STAMPS', stamps, ['id','y_min','y_max','x_min','x_max'], ['I','I','I','I','I'],reduction_metadata, 
			      './', metadata_name='pyDANDIA_metadata.fits')
stage0.add_image_to_the_metadata('MASTER_MASK', master_mask, reduction_metadata, 
			      './', metadata_name='pyDANDIA_metadata.fits')

stage0.save_the_metadata(reduction_metadata, './', metadata_name='pyDANDIA_metadata.fits')
import pdb; pdb.set_trace()
