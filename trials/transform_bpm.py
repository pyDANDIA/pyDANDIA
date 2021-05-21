from astropy.io import fits
from os import path
from sys import argv
import glob

def walk_red_dirs(params):

    red_dir_list = get_red_dir_list(params)

    for red_dir in red_dir_list:
        transform_bpms_for_dataset(red_dir)

def get_red_dir_list(params):

    target_sources = [ 'Gaia', 'ZTF', 'MOA', 'ROME' ]
    red_dir_list = []
    for survey in target_sources:
        dirs = glob.glob(path.join(params['top_dir'],survey+'*'))
        for red_dir in dirs:
            if path.isdir(red_dir) and path.isdir(path.join(red_dir,'data')):
                red_dir_list.append(red_dir)

    return red_dir_list

def transform_bpms_for_dataset(red_dir):

    images = glob.glob(path.join(red_dir,'data','*.fits'))
    for image in images:
        transform_bpm(image)

def transform_bpm(frame):
    hdr = fits.getheader(frame)
    if 'WCSERR' not in hdr.keys() or hdr['WCSERR'] != 0:
        hdu = fits.open(frame)
        for i in range(1,len(hdu),1):
            if 'BPM' in hdu[i].header['EXTNAME'] or 'ERR' in hdu[i].header['EXTNAME']:
                hdu[i].data = hdu[i].data[::-1,::-1]
                print(' -> Transformed BPM (extn '+str(i)+') for '+path.basename(frame))
        hdu.writeto(frame, overwrite=True)
        hdu.close()

if __name__ == '__main__':
    params = {}
    if len(argv) == 1:
        params['top_dir'] = input('Please enter the path to the top-level reduction directory: ')
    else:
        params['top_dir'] = argv[1]

    walk_red_dirs(params)
