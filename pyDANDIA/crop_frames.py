from os import path, mkdir
from sys import argv
from astropy.io import fits
import glob
import shutil

def run_crop_dataset():
    """Function to crop a directory of images according to the parameters
    provided"""

    params = get_args()

    params = review_data(params)

    params = calc_crop_limits(params)

    crop_images(params)

    print('Cropped '+str(len(params['image_list']))+' images output to '+\
                params['red_dir'])
    print('Fullframe images backed up to '+params['bkup_dir'])

def get_args():
    """Function to gather the parameters required to crop a dataset of images"""

    params = {}
    if len(argv) == 1:
        params['red_dir'] = input('Please enter the path to the directory of images: ')
        params['xcentre'] = input('Please input the central position around which to crop in X [pixels]: ')
        params['ycentre'] = input('Please input the central position around which to crop in Y [pixels]: ')
        params['xwidth'] = input('Please input the full width of the box to crop to in X-direction [pixels]: ')
        params['ywidth'] = input('Please input the full width of the box to crop to in Y-direction [pixels]: ')
    else:
        params['red_dir'] = argv[1]
        params['xcentre'] = float(argv[2])
        params['ycentre'] = float(argv[3])
        params['xwidth'] = float(argv[4])
        params['ywidth'] = float(argv[5])

    print(params)
    return params

def review_data(params):
    """Function to perform sanity checks to make sure data is available to
    process, and to add a list of images to the parameters dictionary"""

    if path.isdir(params['red_dir']) == False:
        raise IOError('Cannot find input directory '+params['red_dir'])

    list1 = glob.glob(path.join(params['red_dir'],'*.fits'))
    list2 = glob.glob(path.join(params['red_dir'],'*.fts'))
    list3 = glob.glob(path.join(params['red_dir'],'*.fit'))
    params['image_list'] = list1 + list2 + list3

    if len(params['image_list']) < 1:
        raise IOError('No FITS images found in image directory '+params['red_dir'])

    header = fits.getheader(params['image_list'][0])
    params['NAXIS1'] = float(header['NAXIS1'])
    params['NAXIS2'] = float(header['NAXIS2'])

    if (params['xcentre'] >= params['NAXIS2']) or (params['ycentre'] >= params['NAXIS1']):
        raise ValueError('Requested crop centre ('+str(params['xcentre'])+\
                        ','+str(params['ycentre'])+\
                        ') is outside the image limits of ('+\
                        str(params['NAXIS2'])+','+str(params['NAXIS1'])+')')
                        
    params['bkup_dir'] = path.join(params['red_dir'],'fullframe')
    if path.isdir(params['bkup_dir']) == False:
        mkdir(params['bkup_dir'])

    return params

def calc_crop_limits(params):
    """Function to calculate the crop limits"""

    x_half_width = params['xwidth']/2.0
    y_half_width = params['ywidth']/2.0

    params['xmin'] = int( max( (params['xcentre'] - x_half_width), 0.0 ) )
    params['xmax'] = int( min( (params['xcentre'] + x_half_width), params['NAXIS2'] ) )
    params['ymin'] = int( max( (params['ycentre'] - y_half_width), 0.0 ) )
    params['ymax'] = int( min( (params['ycentre'] + y_half_width), params['NAXIS1'] ) )

    print('Calculated crop limits: ')
    print('X: '+str(params['xmin'])+':'+str(params['xmax']))
    print('Y: '+str(params['ymin'])+':'+str(params['ymax']))

    return params

def crop_images(params):

    for image in params['image_list']:

        hdu = fits.open(image)

        data = hdu[0].data[params['ymin']:params['ymax'], params['xmin']:params['xmax']]

        hdu_out = fits.PrimaryHDU(data=data, header=hdu[0].header)

        bkup_file = path.join(params['bkup_dir'], path.basename(image))
        shutil.move(image, bkup_file)

        hdu_out.writeto(image)


if __name__ == '__main__':
    run_crop_dataset()
