from os import path, rename
from sys import argv
from pyDANDIA import image_handling
from pyDANDIA import logs
from astropy.io import fits

def unmask(params):

    log = logs.start_stage_log(params['red_dir'], 'assign_bpm')

    # If there is a preexisting master_mask rename it
    mask_file_path = path.join(params['red_dir'],'ref/','master_mask.fits')
    if path.isfile(mask_file_path):
        bkup_mask_path = path.join(params['red_dir'],'ref/','default_master_mask.fits')
        rename(mask_file_path, bkup_mask_path)
        log.info('Renamed preexisting BPM to '+bkup_mask_path)

    # Read the data from the old mask
    bpm_data = fits.open(bkup_mask_path)[0]
    log.info('Extracted BPM data from reference image '+params['ref_file_path'])

    # Identify the pixels close to the target location and set the mask to zero
    xmin = params['target_x']-params['dx']
    xmax = params['target_x']+params['dx']
    ymin = params['target_y']-params['dy']
    ymax = params['target_y']+params['dy']
    bpm_data[ymin:ymax, xmin:xmax] = 0

    # Output the revised BPM
    master_mask_hdu = fits.PrimaryHDU(bpm_data)
    master_mask_hdu.writeto(mask_file_path, overwrite=True)

    logs.close_log(log)

def get_args():
    params = {}
    if len(argv) == 1:
        params['red_dir'] = input('Please enter the path to the reduction directory: ')
        params['target_x'] = float(input('Please enter the x-centroid of the target: '))
        params['target_y'] = float(input('Please enter the y-centroid of the target: '))
    else:
        params['red_dir'] = argv[1]
        params['target_x'] = float(argv[2])
        params['target_y'] = float(argv[3]])
    return params

if __name__ == '__main__':
    params = get_args()
    unmask(params)
