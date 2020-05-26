import os
import sys
from pyDANDIA import metadata
from astropy.io import fits
import numpy as np

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

    for i, image in enumerate(reduction_metadata.headers_summary[1]['IMAGES']):
        hdr = fits.getheader(os.path.join(red_dir, 'data', image))
        reduction_metadata.update_a_cell_to_layer('headers_summary', i, col_name, hdr[hdr_key])

    reduction_metadata.save_updated_metadata(red_dir,'pyDANDIA_metadata.fits')

def restore_psf_dimensions_table(red_dir):

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(red_dir, 'pyDANDIA_metadata.fits')

    reduction_metadata.remove_metadata_layer('psf_dimensions', red_dir,
                                                'pyDANDIA_metadata.fits')

    data = []
    for i,factor in enumerate(range(2,5,1)):
        r = input('Please enter the PSF radius for PSF factor '+str(factor)+': ')
        data.append([str(i+1),factor,r])

    reduction_metadata.create_psf_dimensions_layer(np.array(data))
    reduction_metadata.save_updated_metadata(red_dir,'pyDANDIA_metadata.fits')

def edit_image_reduction_status(red_dir):

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(red_dir, 'pyDANDIA_metadata.fits')

    if len(sys.argv) > 2:
        image_name = sys.argv[2]
        status = sys.argv[3]
    else:
        image_name = input('Please enter the name of the image whose status you want to edit: ')
        status = input('Please enter, in list format, the updated status of this image in all stages (e.g. [0,0,0,0,0,0,0,0]): ')

    status_list = status.replace('[','').replace(']','').replace(' ','').split(',')

    if len(status_list) != 8:
        raise IOError('Wrong number of entries for the status of the image in each stage (need 8)')

    idx = np.where(reduction_metadata.reduction_status[1]['IMAGES'] == image_name)[0]

    for i in range(0,8,1):
        col_name = 'STAGE_'+str(i)
        reduction_metadata.update_a_cell_to_layer('reduction_status', idx, col_name, status_list[i])

    reduction_metadata.save_updated_metadata(red_dir,'pyDANDIA_metadata.fits')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        red_dir = input('Please enter the path to the reduction directory: ')
    else:
        red_dir = sys.argv[1]

    #modify_red_status_table(red_dir)
    #modify_reduction_parameters(red_dir)
    #modify_headers_summary(red_dir)
    #restore_psf_dimensions_table(red_dir)
    edit_image_reduction_status(red_dir)
