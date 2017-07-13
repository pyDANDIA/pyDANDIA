import sys
import matplotlib.pyplot as plt

sys.path.append('../pyDANDIA/')
import stage0



reduction_metadata = stage0.create_or_load_the_reduction_metadata('./data/', metadata_name='pyDANDIA_metadata.fits', verbose=True)

data = stage0.find_images_to_process('./data/',verbose=True)

already_reduce_data = stage0.find_images_already_process(reduction_metadata, verbose=True)

new_data = stage0.remove_images_already_process(data, already_reduce_data, verbose=True)

for data in new_data:
	open_image = stage0.open_an_image('./data/',data, verbose=True)
	master_mask = stage0.construct_the_pixel_mask(open_image)
	stamps = stage0.construct_the_stamps(open_image)
	plt.imshow(open_image[0].data[stamps[4,1]:stamps[4,2],stamps[4,3]:stamps[4,4]])
	plt.show()
	import pdb; pdb.set_trace()

