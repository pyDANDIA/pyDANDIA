from sys import argv
from os import path
import numpy as np
from pyDANDIA import hd5_utils
from pyDANDIA import logs
from pyDANDIA import metadata
from pyDANDIA import pipeline_setup
from pyDANDIA import normalize_photometry
from pyDANDIA import crossmatch
from pyDANDIA import field_photometry
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def calc_field_rms():

    params = get_args()

    log = logs.start_stage_log( params['red_dir'], 'field_rms' )

    # Crossmatch table provides information on the filter used for each image
    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(params['crossmatch_file'],log=log)
    log.info('Loaded crossmatch table for the field')

    filter_list = np.unique(xmatch.images['filter'].data)
    log.info('Identified list of filters to process: '+repr(filter_list))

    log.info('Loading the timeseries photometry...')
    phot_data = hd5_utils.read_phot_from_hd5_file(params['phot_file'], return_type='array')
    log.info('-> Completed photometry load')

    # By default, select the columns of photometry that have been normalized:
    (mag_col, mag_err_col) = field_photometry.get_field_photometry_columns('normalized')
    qc_col = 16

    # Plot a separate RMS diagram for each filter
    for filter in filter_list:
        image_index = np.where(xmatch.images['filter'] == filter)[0]
        phot_data_filter = phot_data[:,image_index,:]
        normalize_photometry.plot_multisite_rms(params, phot_data_filter, mag_col, mag_err_col, qc_col,
                                                params['plot_file'], log)

    logs.close_log(log)

def get_args():

    params = {}

    if len(argv) == 1:

        params['red_dir'] = input('Please enter the path to the top-level data directory: ')
        params['field_name'] = input('Please enter the name of the field: ')
        params['quadrant'] = input('Please enter the index number of the quadrant to analyse: ')

    else:

        params['red_dir'] = argv[1]
        params['field_name'] = argv[2]
        params['quadrant'] = argv[3]

    params['crossmatch_file'] = path.join(params['red_dir'],
                            params['field_name']+'_field_crossmatch.fits')
    params['phot_file'] = path.join(params['red_dir'],
                            params['field_name']+'_quad'+params['quadrant']
                                +'_photometry.hdf5')
    params['plot_file'] = params['field_name']+'_quad'+params['quadrant'] \
                                +'_rms_postnorm.png'
    params['text_file'] = params['field_name']+'_quad'+params['quadrant'] \
                                +'_rms_postnorm.txt'

    return params


if __name__ == '__main__':
    calc_field_rms()
