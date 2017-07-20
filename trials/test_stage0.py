import sys
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits

sys.path.append('../pyDANDIA/')
import stage0

data = fits.open('./data/lsc1m005-fl15-20170614-0130-e91.fits')

aa = stage0.read_the_config_file('../Config/')
aa['pixel_scale'] = 0.389
stamps =  stage0.construct_the_stamps(data,arcseconds_stamp_size=(60,60),pixel_scale=aa['pixel_scale'])

#import pdb; pdb.set_trace()

#reduction_metadata = stage0.create_or_load_the_reduction_metadata('./', metadata_name='pyDANDIA_metadata.fits', verbose=True)

data = stage0.find_images_to_process('./data/',verbose=True)

already_reduce_data = stage0.find_images_already_process(reduction_metadata, verbose=True)

new_data = stage0.remove_images_already_process(data, already_reduce_data, verbose=True)

for data in new_data:
	open_image = stage0.open_an_image('./data/',data, verbose=True)
	master_mask = stage0.construct_the_pixel_mask(open_image)
	stamps = stage0.construct_the_stamps(open_image)

images = np.c_[new_data,np.arange(0,len(new_data))]

stage0.add_table_to_the_metadata('IMAGES', images, ['NAMES','ID'], ['100A','I'],reduction_metadata, 
			      './', metadata_name='pyDANDIA_metadata.fits')
stage0.add_table_to_the_metadata('STAMPS', stamps, ['id','y_min','y_max','x_min','x_max'], ['I','I','I','I','I'],reduction_metadata, 
			      './', metadata_name='pyDANDIA_metadata.fits')
stage0.add_image_to_the_metadata('MASTER_MASK', master_mask, reduction_metadata, 
			      './', metadata_name='pyDANDIA_metadata.fits')

stage0.save_the_metadata(reduction_metadata, './', metadata_name='pyDANDIA_metadata.fits')
import pdb; pdb.set_trace()
