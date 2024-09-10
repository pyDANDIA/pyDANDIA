from os import path, rename
from sys import argv
from pyDANDIA import pipeline_setup
from pyDANDIA import logs
from pyDANDIA import metadata
from pyDANDIA import hd5_utils
from pyDANDIA import calibrate_photometry
from pyDANDIA import photometry
import astropy.units as u
from astropy.table import Table, Column
import numpy as np

VERSION = 'recalibrate_photometry_1.0'

def apply_photometric_calibration(setup, params):

    log = logs.start_stage_log( setup.red_dir, 're_phot_calib', version=VERSION )

    # Read the current photometric calibration parameters from the metadata
    reduction_metadata = fetch_metadata(setup,log)
    (fit, covar_fit) = fetch_phot_calib(reduction_metadata,log)

    # Read in the timeseries photometry
    phot_file = path.join(setup.red_dir,'photometry.hdf5')
    phot_data = hd5_utils.read_phot_from_hd5_file(phot_file, return_type='array')
    log.info('Loaded dataset photometry and metadata')

    # Apply the photometric calibration to the timeseries photometry
    phot_data = recalc_calib_phot(phot_data, fit, covar_fit, log)

    # Output the updated photometry
    output_revised_photometry(setup, phot_data, log)

    logs.close_log(log)

    status = 'OK'
    report = 'Completed successfully'

    return status, report

def recalc_calib_phot(phot_data, fit, covar_fit, log):
    """Function applies the photometric calibration to the timeseries photometry
    from all images in the dataset"""

    # Columns of the instrumental magnitudes in the HDF photometry table:
    mag_col = 11
    mag_err_col = 12
    cal_mag_col = 13
    cal_merr_col = 14
    cal_flux_col = 17
    cal_flux_err_col = 18
    nstars = phot_data.shape[0]
    nimages = phot_data.shape[1]

    # Loop over all images in the dataset.  It would be faster to perform this
    # operation on the entire photometry array, but I've implemented it this
    # way in order to re-use critical code from calibrate_photometry.
    log.info('Applying photometric calibration to timeseries data:')
    for i in range(0,phot_data.shape[1],1):
        log.info('-> Image '+str(i)+' of '+str(nimages))

        # Extract 2D arrays of the photometry of all stars in all frames,
        # and mask invalid values
        valid = np.logical_and(phot_data[:,i,mag_col] > 0.0, \
                                        phot_data[:,i,mag_err_col] > 0.0)
        invalid = np.invert(valid)

        star_cat = Table([Column(name='mag', data=phot_data[:,i,mag_col]),
                          Column(name='mag_err', data=phot_data[:,i,mag_err_col]),
                          Column(name='cal_ref_mag', data=np.zeros(nstars)),
                          Column(name='cal_ref_mag_err', data=np.zeros(nstars)),
                          Column(name='cal_ref_flux', data=np.zeros(nstars)),
                          Column(name='cal_ref_flux_err', data=np.zeros(nstars))])

        star_cat = calibrate_photometry.calc_calibrated_mags(fit, covar_fit,
                                                             star_cat, log)
        (star_cat['cal_ref_flux'], star_cat['cal_ref_flux_err']) = photometry.convert_mag_to_flux(star_cat['cal_ref_mag'],
                                                                                                  star_cat['cal_ref_mag_err'])
        phot_data[valid,i,cal_mag_col] = star_cat['cal_ref_mag'][valid]
        phot_data[valid,i,cal_merr_col] = star_cat['cal_ref_mag_err'][valid]
        phot_data[valid,i,cal_flux_col] = star_cat['cal_ref_flux'][valid]
        phot_data[valid,i,cal_flux_err_col] = star_cat['cal_ref_flux_err'][valid]

    return phot_data

def fetch_phot_calib(reduction_metadata,log):
    """Retrieve the photometric calibration from the metadata"""

    fit = [reduction_metadata.phot_calib[1]['a0'][0],
            reduction_metadata.phot_calib[1]['a1'][0]]

    covar_fit = np.array([ [reduction_metadata.phot_calib[1]['c0'][0],
                            reduction_metadata.phot_calib[1]['c1'][0]],
                           [reduction_metadata.phot_calib[1]['c2'][0],
                            reduction_metadata.phot_calib[1]['c3'][0]] ])

    log.info('Loaded existing photometric calibration parameters: ')
    log.info('Fit: '+repr(fit))
    log.info('Covarience of fit: '+repr(covar_fit))

    return fit, covar_fit

def fetch_metadata(setup,log):
    """Function to extract the information necessary for the photometric
    calibration from a metadata file"""

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'data_architecture' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                               'pyDANDIA_metadata.fits',
                                              'reduction_parameters' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'phot_calib' )

    return reduction_metadata

def output_revised_photometry(setup, photometry, log):

    # Back up the older photometry file for now
    phot_file_name = path.join(setup.red_dir,'photometry.hdf5')
    bkup_file_name = path.join(setup.red_dir,'photometry_stage6.hdf5')
    if path.isfile(phot_file_name):
        rename(phot_file_name, bkup_file_name)

    # Output file with additional columns:
    hd5_utils.write_phot_hd5(setup,photometry,log=log)

def get_args():

    params = { 'red_dir': '',
               'metadata': '',
               'log_dir': '',
               'pipeline_config_dir': '',
               'software_dir': '',
               'verbosity': '' }

    if len(argv) > 1:
        params['red_dir'] = argv[1]
        params['log_dir'] = argv[2]
    else:
        params['red_dir'] = input('Please enter the path to the reduction directory: ')
        params['log_dir'] = input('Please enter the path to the log directory: ')

    setup = pipeline_setup.pipeline_setup(params)

    return setup, params


if __name__ == '__main__':
    (setup, params) = get_args()
    (status, report) = apply_photometric_calibration(setup, params)
