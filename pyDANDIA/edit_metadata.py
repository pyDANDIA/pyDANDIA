import os
import sys
from pyDANDIA import metadata
from astropy.io import fits
import numpy as np
import copy
import shutil

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

def update_software_table(red_dir):

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(red_dir, 'pyDANDIA_metadata.fits')

    reduction_metadata.update_a_cell_to_layer('software', 0, 'stage3_version', 'pyDANDIA_stage3_v1.0.0')
    reduction_metadata.update_a_cell_to_layer('software', 0, 'stage6_version', 'pyDANDIA_stage6_v1.0.0')

    reduction_metadata.save_updated_metadata(red_dir,'pyDANDIA_metadata.fits')

def remove_table(red_dir):

    table_names = input('Please enter the name of the table extension to remove (comma-separated, no spaces): ')

    table_names = table_names.split(',')

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(red_dir, 'pyDANDIA_metadata.fits')

    meta2 = copy.deepcopy(reduction_metadata)
    shutil.move(os.path.join(red_dir, 'pyDANDIA_metadata.fits'),os.path.join(red_dir, 'pyDANDIA_metadata.fits.old'))

    for table in table_names:
        meta2.remove_metadata_layer(table, red_dir, 'pyDANDIA_metadata.fits')

    # This re-writes the file because the metadata's built-in function
    # does not remove a table - it can only update the contents of an existing
    # table
    hdulist = [fits.PrimaryHDU()]
    all_layers = meta2.__dict__.keys()
    for key_layer in all_layers:
        layer = getattr(meta2, key_layer)
        if layer != [None, None]:
            update_layer = fits.BinTableHDU(layer[1], header=layer[0])
            update_layer.name = update_layer.header['name']
            hdulist.append(update_layer)

    hdulist = fits.HDUList(hdulist)
    hdulist.writeto(os.path.join(red_dir, 'pyDANDIA_metadata.fits'),
                     overwrite=True)

    print('Output revised metadata')

def change_reduction_dir(red_dir):

    new_red_dir = input('Please enter the new reduction directory path: ')

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(red_dir, 'pyDANDIA_metadata.fits')

    print(reduction_metadata.data_architecture[1])
    reduction_metadata.update_a_cell_to_layer('data_architecture', 0, 'OUTPUT_DIRECTORY', new_red_dir)
    reduction_metadata.update_a_cell_to_layer('data_architecture', 0, 'IMAGES_PATH', os.path.join(new_red_dir,'data'))
    reduction_metadata.update_a_cell_to_layer('data_architecture', 0, 'REF_PATH', os.path.join(new_red_dir,'ref'))
    reduction_metadata.update_a_cell_to_layer('data_architecture', 0, 'KERNEL_PATH', os.path.join(new_red_dir,'kernel'))
    reduction_metadata.update_a_cell_to_layer('data_architecture', 0, 'DIFFIM_PATH', os.path.join(new_red_dir,'diffim'))

    reduction_metadata.save_updated_metadata(red_dir,'pyDANDIA_metadata.fits')

if __name__ == '__main__':
    if len(sys.argv) == 1:
        red_dir = input('Please enter the path to the reduction directory: ')
        print("""Main menu:
                Modify reduction status table        1
                Modify reduction parameters          2
                Modify headers summary               3
                Restore PSF dimensions table         4
                Edit image reduction status          5
                Update software table                6
                Remove a table                       7
                Change a reduction directory         8
                Cancel                               Any other key""")
        opt = input('Please select an option: ')
    else:
        red_dir = sys.argv[1]
        opt = sys.argv[2]

    if opt == '1':
        modify_red_status_table(red_dir)
    elif opt == '2':
        modify_reduction_parameters(red_dir)
    elif opt == '3':
        modify_headers_summary(red_dir)
    elif opt == '4':
        restore_psf_dimensions_table(red_dir)
    elif opt == '5':
        edit_image_reduction_status(red_dir)
    elif opt == '6':
        update_software_table(red_dir)
    elif opt == '7':
        remove_table(red_dir)
    elif opt == '8':
        change_reduction_dir(red_dir)
