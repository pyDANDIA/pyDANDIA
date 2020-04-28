import os
import sys
from pyDANDIA import metadata
from astropy.io import fits

def modify_red_status_table(red_dir):

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(red_dir, 'pyDANDIA_metadata.fits')

    n_images = len(reduction_metadata.reduction_status[1]['IMAGES'])

    for i in range(0,n_images,1):
        if '-1' in str(reduction_metadata.reduction_status[1]['STAGE_7'][i]):
            reduction_metadata.update_a_cell_to_layer('reduction_status', i, 'STAGE_6', '-1')

    reduction_metadata.save_updated_metadata(red_dir,'pyDANDIA_metadata.fits')

def modify_reduction_parameters(red_dir):

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(red_dir, 'pyDANDIA_metadata.fits')

    key = input('Enter keyword value to change: ')
    value = input('Enter new value of this keyword: ')

    reduction_metadata.update_a_cell_to_layer('reduction_parameters', 0, key, str(value))

    reduction_metadata.save_updated_metadata(red_dir,'pyDANDIA_metadata.fits')

def modify_headers_summary(red_dir):

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(red_dir, 'pyDANDIA_metadata.fits')

    col_name = input('Enter name of column to update: ')
    hdr_key = input('Enter name of FITS header keyword to update the column with: ')

    for i, image in reduction_metadata.headers_summary[1]['IMAGES']:
        hdr = fits.getheader(os.path.join(red_dir, 'data', image))
        reduction_metadata.update_a_cell_to_layer('headers_summary', i, col_name, hdr[hdr_key])

    reduction_metadata.save_updated_metadata(red_dir,'pyDANDIA_metadata.fits')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        red_dir = input('Please enter the path to the reduction directory: ')
    else:
        red_dir = sys.argv[1]

    #modify_red_status_table(red_dir)
    modify_reduction_parameters(red_dir)
