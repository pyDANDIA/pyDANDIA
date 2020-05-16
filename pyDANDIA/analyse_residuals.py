from sys import argv
from os import getcwd, path, remove, environ
import numpy as np
from pyDANDIA import  phot_db
from pyDANDIA import  hd5_utils
from pyDANDIA import  pipeline_setup
from pyDANDIA import  metadata
from pyDANDIA import  logs
from pyDANDIA import pipeline_control
from pyDANDIA import stage3_db_ingest
from pyDANDIA import match_utils
from scipy import optimize
import matplotlib.pyplot as plt
from pyDANDIA import  hd5_utils
from astropy import table

VERSION = 'analyse_residuals_v0.1.0'

def run_residual_analyses(setup):
    """Function to analyse the photometric timeseries residuals for a given
    dataset and re-weight the photometric uncertainties for all measurements
    from each image based on the scatter in the ensamble residuals."""

    log = logs.start_pipeline_log(setup.red_dir, 'analyse_residuals',
                                  version=VERSION)

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(metadata_directory=setup.red_dir,
                                         metadata_name='pyDANDIA_metadata.fits')

    photometry_file = path.join(setup.red_dir, 'photometry.hdf5')

    if path.isfile(photometry_file) == False:
        report = 'No timeseries photometry found at '+photometry_file+\
                    ' so no residuals analysis performed'

        log.info('Analyse_residuals: '+report)
        logs.close_log(log)
        status = 'OK'

        return status, report

    photometry = hd5_utils.load_dataset_timeseries_photometry(setup,log,27)

    mean_mags = calc_star_mean_mags(photometry,log)

    residuals = calc_lightcurve_residuals(photometry,mean_mags,log)

    phot_scatter = calc_image_photometric_scatter(reduction_metadata,residuals,log)

    reduction_metadata.create_a_new_layer_from_table('phot_scatter', phot_scatter)

    photometry = update_photometric_uncertainties(photometry, phot_scatter, log)

    hd5_utils.write_phot_hd5(setup, photometry,log=log)

def calc_star_mean_mags(photometry,log):
    """Function to calculate the mean magnitude for all stars in the dataset
    from their timeseries photometry, weighted by the photometric uncertainties."""

    mean_mags = np.zeros( [photometry.shape[0],2] )

    weights = 1.0 / (photometry[:,:,25]*photometry[:,:,25])
    mean_mags[:,0] = (photometry[:,:,24] * weights).sum(axis=1) / weights.sum(axis=1)
    mean_mags[:,1] = 1.0 / np.sqrt( weights.sum(axis=1) )

    log.info('Calculated the weighted mean magnitude of '+str(photometry.shape[0])+' stars')

    return mean_mags

def calc_lightcurve_residuals(photometry,mean_mags,log):
    """Function to subtract the weighted mean magnitude of each star from its
    lightcurve to produce the photometric residuals."""

    residuals = photometry[:,:,24:25]
    residuals[:,:,0] -= mean_mags[:,0]
    residuals[:,:,1] = np.sqrt( photometry[:,:,25]*photometry[:,:,25] + \
                                    mean_mags[:,1]*mean_mags[:,1] )

    log.info('Calculated the lightcurves residuals')

    return residuals

def calc_image_photometric_scatter(reduction_metadata,residuals,log):
    """Function to calculate the weighted mean photometric residual per image"""

    weights = 1.0 / (residuals[:,:,1]*residuals[:,:,1])
    scatter = (residuals[:,:,0] * weights).sum(axis=0) / weights.sum(axis=0)

    log.info('Calculated photometric scatter per image [mag]: ')
    log.info(repr(phot_scatter))

    if len(reduction_metadata.images_stats[1]['IM_NAME']) == len(scatter):
        phot_scatter = Table( [ Column(name='IMAGES', data=reduction_metadata.images_stats[1]['IM_NAME'], dtype='str'),
                                Column(name='SCATTER', data=scatter, dtype='float') ] )
    else:
        raise IOError('Photometry array has data for a different number ('+str(len(scatter))+\
                        ') of images than are listed in the metadata ('+\
                        str(len(reduction_metadata.images_stats[1]['IM_NAME']))+')')

    return phot_scatter

def update_photometric_uncertainties(photometry, phot_scatter, log):
    """Function to re-calculate the lightcurve photometric uncertainties for all
    stars, taking into account the scatter per image"""

    # Photometry taken from the cross-calibrated magnitudes column
    photometry[:,:,26] = photometry[:,:,24]
    photometry[:,:,27] = np.sqrt( photometry[:,:,25]*photometry[:,:,25] + \
                                    phot_scatter['SCATTER']*phot_scatter['SCATTER'] )

    return photometry
