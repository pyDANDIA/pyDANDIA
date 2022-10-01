from os import path, rename
from sys import argv
from pyDANDIA import image_handling
from pyDANDIA import logs
from astropy.io import fits

def use_bpm_from_ref_image(params):
    """Function replaces the default master_mask.fits Bad Pixel Mask with
    the BPM from the reference image."""

    log = logs.start_stage_log(params['red_dir'], 'assign_bpm')

    # If there is a preexisting master_mask rename it
    mask_file_path = path.join(params['red_dir'],'ref/','master_mask.fits')
    if path.isfile(mask_file_path):
        bkup_mask_path = path.join(params['red_dir'],'ref/','default_master_mask.fits')
        rename(mask_file_path, bkup_mask_path)
        log.info('Renamed preexisting BPM to '+bkup_mask_path)

    # Extract the BPM data from the reference image
    ref_structure = image_handling.determine_image_struture(params['ref_file_path'], log=log)
    bpm_data = fits.open(params['ref_file_path'])[ref_structure['bpm']].data.astype(float)
    log.info('Extracted BPM data from reference image '+params['ref_file_path'])

    # Save the new BPM
    master_mask_hdu = fits.PrimaryHDU(bpm_data)
    master_mask_hdu.writeto(mask_file_path, overwrite=True)
    log.info('Output reference BPM to '+mask_file_path)

    logs.close_log(log)

def get_args():

    params = {}
    if len(argv) == 1:
        params['ref_file_path'] = input('Please enter the path to the reference image: ')
    else:
        params['ref_file_path'] = argv[1]
    params['red_dir'] = path.dirname(path.join(path.dirname(params['ref_file_path']),'../'))

    return params

if __name__ == '__main__':
    params = get_args()
    use_bpm_from_ref_image(params)
