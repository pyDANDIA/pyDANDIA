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
import pdb; pdb.set_trace()

reduction_metadata = stage0.create_or_load_the_reduction_metadata('./', metadata_name='pyDANDIA_metadata.fits', verbose=True)


stage0.update_reduction_metadata_with_config_file(reduction_metadata, aa)
import pdb; pdb.set_trace()
data = stage0.find_all_images(reduction_metadata, './data/',verbose=True)

new_images = stage0.find_images_need_to_be_process(reduction_metadata, data,verbose=True)
aa = stage0.read_the_config_file('../Config/')


for data in new_images:
	start = time.time()
	open_image = stage0.open_an_image( reduction_metadata.data_architecture[1]['IMAGES_PATH'][0],data, verbose=True)
	end = time.time()
	print end-start
	if reduction_metadata.stamps == [None,None]:
			stamps = stage0.construct_the_stamps(open_image, pixel_scale = 0.389)
			end = time.time()
			print end-start
			data_structure = [['ID','Y_MIN','Y_MAX','X_MIN','X_MAX'],
			 		 ]
			reduction_metadata.create_a_new_layer('stamps', data_structure, data_columns = stamps)
			end = time.time()
			print end-start
			

	master_mask = stage0.construct_the_pixel_mask(open_image)
	end = time.time()
	print end-start	
	
reduction_metadata.create_reduction_status_layer(new_images)
end = time.time()
print end-start
reduction_metadata.save_updated_metadata( './', metadata_name='pyDANDIA_metadata.fits')
end = time.time()
print end-start
