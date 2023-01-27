import os
import sys
from pyDANDIA import pipeline_setup
from pyDANDIA import logs
from pyDANDIA import metadata
import astropy.units as u
from astropy.table import Table, Column
import numpy as np

VERSION = 'recalibrate_photometry_1.0'

def apply_photometric_calibration(setup, params):

    log = logs.start_stage_log( setup.red_dir, 're_phot_calib', version=VERSION )

    (reduction_metadata, params) = fetch_metadata(setup,params,log)

    (fit, covar_fit) = fetch_phot_calib(reduction_metadata,params,log)

    phot_file = path.join(setup.red_dir,'photometry.hdf5')
    photometry = hd5_utils.read_phot_from_hd5_file(phot_file, return_type='array')
    log.info('Loaded dataset photometry and metadata')


    logs.close_log(log)

    status = 'OK'
    report = 'Completed successfully'

    return status, report

def recalc_calib_phot(photometry, fit, covar_fit, log):
    """Function applies the photometric calibration to the timeseries photometry
    from all images in the dataset"""

    # Columns of the instrumental magnitudes in the HDF photometry table:
    mag_col = 11
    mag_err_col = 12
    cal_mag_col = 13
    cal_merr_col = 14
    cal_flux_col = 17
    cal_flux_err_col = 18
    nstars = photometry.shape[0]

    # Loop over all images in the dataset.  It would be faster to perform this
    # operation on the entire photometry array, but I've implemented it this
    # way in order to re-use critical code from calibrate_photometry.
    for i in range(0,photometry.shape[1],1):
        # Extract 2D arrays of the photometry of all stars in all frames,
        # and mask invalid values
        mask = np.invert(np.logical_and(mags > 0.0, errs > 0.0))

        star_cat = Table([Column(name='mag', data=photometry[:,i,mag_col]),
                          Column(name='mag_err', data=photometry[:,i,mag_err_col]),
                          Column(name='cal_ref_mag', data=np.zeros(nstars)),
                          Column(name='cal_ref_mag_err', data=np.zeros(nstars)),
                          Column(name='cal_ref_flux', data=np.zeros(nstars)),
                          Column(name='cal_ref_flux_err', data=np.zeros(nstars))])

        star_cat = calibrate_photometry.calc_calibrated_mags(fit, covar_fit,
                                                            star_cat, log)
        (star_cat['cal_ref_flux'], star_cat['cal_ref_flux_err']) = photometry.convert_mag_to_flux(star_cat['cal_ref_mag'],
                                                                                                  star_cat['cal_ref_mag_err'])

        photometry[mask,i,cal_mag_col] = star_cat['cal_ref_mag'][mask]
        photometry[mask,i,cal_merr_col] = star_cat['cal_ref_mag_err'][mask]
        photometry[mask,i,cal_flux_col] = star_cat['cal_ref_flux'][mask]
        photometry[mask,i,cal_flux_err_col] = star_cat['cal_ref_flux_err'][mask]

    return photometry

def fetch_phot_calib(reduction_metadata,params,log):
    """Retrieve the photometric calibration from the metadata"""

    fit = [reduction_metadata.phot_calib[1]['a0'][0],
            reduction_metadata.phot_calib[1]['a1'][0]]

    covar_fit = np.array([ [reduction_metadata.phot_calib[1]['c0'][0],
                            reduction_metadata.phot_calib[1]['c1'][0],
                            reduction_metadata.phot_calib[1]['c2'][0],
                            reduction_metadata.phot_calib[1]['c3'][0]] ])

    log.info('Loaded existing photometric calibration parameters: ')
    log.info('Fit: '+repr(fit))
    log.info('Covarience of fit: '+repr(covar_fit))

    return fit, covar_fit

def fetch_metadata(setup,params,log):
    """Function to extract the information necessary for the photometric
    calibration from a metadata file, adding information to the params
    dictionary"""

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              params['metadata'],
                                              'data_architecture' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                               params['metadata'],
                                              'reduction_parameters' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              params['metadata'],
                                              'phot_calib' )

    return reduction_metadata, params


def get_args():

    params = { 'red_dir': '',
               'metadata': '',
               'log_dir': '',
               'pipeline_config_dir': '',
               'software_dir': '',
               'verbosity': '' }

    if len(sys.argv) > 1:
        params['red_dir'] = sys.argv[1]
        params['metadata'] = sys.argv[2]
        params['log_dir'] = sys.argv[3]
    else:
        params['red_dir'] = input('Please enter the path to the reduction directory: ')
        params['metadata'] = input('Please enter the name of the metadata file: ')
        params['log_dir'] = input('Please enter the path to the log directory: ')

    setup = pipeline_setup.pipeline_setup(params)

    return params, setup


if __name__ == '__main__':
    (params, setup) = get_args()
    (status, report) = apply_photometric_calibration(setup, params)
