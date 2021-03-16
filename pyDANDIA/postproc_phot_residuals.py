from os import path
from sys import argv
import numpy as np
from pyDANDIA import crossmatch
from pyDANDIA import hd5_utils
from pyDANDIA import logs
from pyDANDIA import metadata
from pyDANDIA import plot_rms

def run_postproc():
    """Driver function for post-processing:
    Assessment of photometric residuals and uncertainties
    """
    params = get_args()

    log = logs.start_stage_log( params['log_dir'], 'postproc_phot' )

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(params['red_dir'], 'pyDANDIA_metadata.fits')
    phot_file = path.join(params['red_dir'],'photometry.hdf5')
    photometry = hd5_utils.read_phot_from_hd5_file(phot_file, return_type='array')
    log.info('Loaded dataset photometry and metadata')

    # Grow photometry array to allow additional columns for corrected mags
    photometry = grow_photometry_array(photometry,log)

    # Calculate mean_mag, RMS for all stars
    phot_stats = phot_rms.calc_mean_rms_mag(photometry,log,use_calib_mag=True)

    # Calculate photometric residuals
    phot_residuals = calc_phot_residuals(photometry, phot_stats, log)

    # Calculate mean residual per image
    image_residuals = calc_image_residuals(phot_residuals,log)
    
    # Compensate for mean residual per image and output to photometry array

    # Re-calculate mean_mag, RMS for all stars, using corrected magnitudes
    phot_stats = phot_rms.calc_mean_rms_mag(photometry,log,use_calib_mag=True)

    # Re-calculate photometric residuals
    phot_residuals = calc_phot_residuals(photometry, phot_stats, log)

    # Calculate photometric scatter per image

    # Factor photometric scatter into photometric residuals

    log.info('Post-processing: complete')

    logs.close_log(log)

def get_args():

    params = {}
    if len(argv) == 1:
        params['red_dir'] = input('Please enter the path to the datasets reduction directory: ')
    else:
        params['red_dir'] = argv[1]

    params['log_dir'] = params['red_dir']

    return params

def grow_photometry_array(photometry,log):

    new_photometry = np.zeros((photometry.shape[0], photometry.shape[1], photometry.shape[2]+2))
    new_photometry[:,:,0:photometry.shape[2]] = photometry
    log.info('Added two columns to the photometry array')

    return new_photometry

def calc_phot_residuals(photometry, phot_stats, log, use_calib_mag=True):

    mag_col = 13
    merr_col = 14
    if not use_calib_mag:
        mag_col = 11
        merr_col = 12

    # To subtract a 1D vector from a 2D array, we need to add an axis to ensure
    # the correct Python array handling
    mean_mag = phot_stats[:,0, np.newaxis]
    mean_merr = phot_stats[:,3, np.newaxis]

    phot_residuals = np.zeros((photometry.shape[0],photometry.shape[1],2))

    phot_residuals[:,:,0] = photometry[:,:,mag_col] - mean_mag
    phot_residuals[:,:,1] = np.sqrt( photometry[:,:,merr_col]*photometry[:,:,merr_col] + \
                                    mean_merr*mean_merr )

    log.info('Calculated photometric residuals')

    return phot_residuals

def calc_image_residuals(phot_residuals,log):

    image_residuals = np.zeros((phot_residuals.shape[1],2))

    err_squared_inv = 1.0 / (phot_residuals[:,:,1]*phot_residuals[:,:,1])
    image_residuals[:,0] =  (phot_residuals[:,:,0] * err_squared_inv).sum(axis=0) / (err_squared_inv.sum(axis=0))
    image_residuals[:,1] = 1.0 / (err_squared_inv.sum(axis=0))

    log.info('Calculated weighted mean photometric residual per image')

    return image_residuals
