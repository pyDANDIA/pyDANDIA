from os import path, rename
from sys import argv
import numpy as np
from pyDANDIA import crossmatch
from pyDANDIA import hd5_utils
from pyDANDIA import logs
from pyDANDIA import pipeline_setup
from pyDANDIA import plot_rms
from pyDANDIA import calibrate_photometry
from pyDANDIA import photometry
from pyDANDIA import field_photometry
from pyDANDIA import field_lightcurves
from astropy.table import Table, Column
import matplotlib.pyplot as plt

VERSION = 'star_norm_v0.1'

def run_star_normalization(setup, **params):

    log = logs.start_stage_log( setup.red_dir, 'postproc_star_norm', version=VERSION )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(params['crossmatch_file'],log=log)
    xmatch.id_primary_datasets_per_filter()

    image_sets = xmatch.get_imagesets()
    filter_list = np.unique(xmatch.images['filter'].data)
    (mag_col, mag_err_col) = field_photometry.get_field_photometry_columns('normalized')
    qc_col = 16

    qid = 1

    file_path = path.join(setup.red_dir, params['field_name']\
                    +'_quad'+str(qid)+'_photometry.hdf5')
    quad_phot = hd5_utils.read_phot_from_hd5_file(file_path, return_type='array')

    field_id = 121425
    field_idx = field_id - 1

    lc = field_lightcurves.fetch_field_photometry_for_star_idx(params, field_idx,
                                                                xmatch, quad_phot,
                                                                log)
    logs.close_log(log)

def normalize_star_datasets(xmatch, lc, filter_list, log):
    """Function bins the star's lightcurve in time, and calculates the weighted
    mean magnitude offset of each dataset from the primary reference lightcurve
    in each filter, from the binned data, excluding outliers. """

    for f in filter_list:
        # Fetch the primary reference lightcurve for this star
        pri_ref_code = xmatch.get_dataset_shortcode(xmatch.reference_datasets[f])
        pri_ref_lc = lc[pri_ref_code]

        binned_pri_ref_lc = bin_lc_in_time(lc, bin_width=1.0)

        
def bin_lc_in_time(lc, bin_width=1.0):
    """Function to bin a lightcurve array into time bins, by default 24hrs
    wide.  Function expects a lightcurve in the form of a numpy array with
    columns HJD, mag, mag_error, qc_flag."""

    time_bins = np.arange(lc[:,0].min(), lc[:,0].max(), bin_width)
    time_index = np.digitize(lc[:,0], time_bins)

    binned_lc = np.zeros((len(time_bins),3))
    binned_lc[:,0] = time_bins
    for b in range(0,len(time_bins),1):
        idx = np.where(time_index == b)
        (wmean, werror) = calc_weighted_mean_datapoint(lc[idx,1], lc[idx,2])
        binned_lc[b,1] = wmean
        binned_lc[b,2] = werror

    return binned_lc

def calc_weighted_mean_datapoint(data, errs):
    """Expects input array of the form: columns HJD, mag, mag_error, qc_flag"""

    err_squared_inv = 1.0 / (errs*errs)
    wmean =  (data * err_squared_inv).sum(axis=1) / (err_squared_inv.sum(axis=1))
    werror = np.sqrt( 1.0 / (err_squared_inv.sum(axis=1)) )

    return wmean, werror

def get_args():
    params = {}
    if len(argv) == 1:
        params['crossmatch_file'] = input('Please enter the path to field crossmatch table: ')
        params['red_dir'] = input('Please enter the path to field top level data reduction directory: ')
        params['field_name'] = input('Please enter the name of the field: ')
    else:
        params['crossmatch_file'] = argv[1]
        params['red_dir'] = argv[2]
        params['field_name'] = argv[3]

    params['log_dir'] = path.join(params['red_dir'],'logs')
    setup = pipeline_setup.pipeline_setup(params)

    return setup, params


if __name__ == '__main__':
    (setup, params) = get_args()
    run_star_normalization(setup, **params)
