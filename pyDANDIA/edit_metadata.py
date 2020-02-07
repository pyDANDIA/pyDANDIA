import os
import sys
from pyDANDIA import metadata

def modify_red_status_table(red_dir):

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(red_dir, 'pyDANDIA_metadata.fits')

    n_images = len(reduction_metadata.reduction_status[1]['IMAGES'])

    for i in range(10,n_images-10,1):
        for c in [4,5,6,7]:
            stage_name = 'STAGE_'+str(c)
            reduction_metadata.update_a_cell_to_layer('reduction_status', i, stage_name, '-1')

    reduction_metadata.save_updated_metadata(red_dir,'pyDANDIA_metadata.fits')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        red_dir = input('Please enter the path to the reduction directory: ')
    else:
        red_dir = sys.argv[1]

    modify_red_status_table(red_dir)
