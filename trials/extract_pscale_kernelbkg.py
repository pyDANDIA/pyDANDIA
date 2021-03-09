import glob
from os import path, sep
from sys import argv
import numpy as np
from astropy.table import Table


def check_rome_stamp_position(stamp_idx):
    """Report ROME stamp center in px, also available from metadata directly"""
    """The exact center can be off due to overlap or border offsets"""
    stamp_center = [[505.0,505.0],\
    [1500.0,505.0],\
    [2500.0,505.0],\
    [3543.0,505.0],\
    [505.0,1500.0],\
    [1500.0,1500.0],\
    [2500.0,1500.0],\
    [3543.0,1500.0],\
    [505.0,2500.0],\
    [1500.0,2500.0],\
    [2500.0,2500.0],\
    [3543.0,2500.0],\
    [505.0,3543.0],\
    [1500.0,3543.],\
    [2500.0,3543.0],\
    [3543.0,3543.0]]
    return stamp_center[stamp_idx]

def extract_scale_factor(output_file_location, reduction_field_dir):
    """Extract scale factor for stamps"""

    output_table = Table(names=('pscale', 'kernel_bkg', 'stamp_center_x',\
                                'stamp_center_y','stamp_index','filename','dataset' ), \
                                dtype=('f4','f4','f4','f4', 'i4', 'S','S'))

    print('Extract scale factor from kernel'+str(reduction_field_dir))
    
    kernel_stamps = glob.glob(path.join(reduction_field_dir,\
                                        'R*p/*kernel*/*/kernel_stamp*.npy'))
    if kernel_stamps == []:
        kernel_stamps = glob.glob(path.join(reduction_field_dir,\
                                           '*kernel*/*/kernel_stamp*.npy'))
    if kernel_stamps == []:
        print('No kernel stamp subdirectories found, check directories please.')
        return None

    for kernel in kernel_stamps:
        print('processing ', kernel)
        image_name = path.normpath(kernel).split(sep)[-2]
        dataset_name = path.normpath(kernel).split(sep)[-3]
        index = path.split(kernel)
        print(image_name)
        if not path.exists(kernel):
            print('No kernel solution found.')
        else:
            kernel_data = np.load(kernel,allow_pickle = True)
            stamp_idx = int(path.split(kernel)[1].split('_')[-1].replace('.npy',''))               
            x_center_stamp, y_center_stamp = check_rome_stamp_position(stamp_idx)
            pscale, bkg = kernel_data[0].sum(), kernel_data[1]
            output_table.add_row((pscale, bkg, x_center_stamp, \
                                  y_center_stamp, stamp_idx, image_name, dataset_name))
    output_table.write(output_file_location, format='fits', overwrite = False)


if __name__ == '__main__':
    if len(argv) == 3:
        field_dir = argv[1]
        output_file = argv[2]
        extract_scale_factor(output_file, field_dir)
    else:
        field_dir = input('Please enter the path to the field reduction directory, e.g. /data/ROME-FIELD-01: ')
        output_file = input('Please enter the output location e.g. /data/ROME-FIELD-01/pscale_bkg_table.fits: ')
        extract_scale_factor(output_file, field_dir)

