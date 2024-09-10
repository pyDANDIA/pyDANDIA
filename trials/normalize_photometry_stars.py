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
from pyDANDIA import normalize_photometry
from astropy.table import Table, Column
import matplotlib.pyplot as plt
from skimage.measure import ransac, LineModelND
import copy

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

    # For testing purposes only:
    qid = 1

    log.info('Loading timeseries photometry...')
    file_path = path.join(setup.red_dir, params['field_name']\
                    +'_quad'+str(qid)+'_photometry.hdf5')
    quad_phot = hd5_utils.read_phot_from_hd5_file(file_path, return_type='array')
    log.info('...completed')

    # Create a holding array for the output measured magnitude offset per dataset.
    xmatch.create_normalizations_table()

    # For testing purposes only:
    field_id = 215316
    field_idx = field_id - 1

    # The normalization process compares the photometry measurements of a given
    # star from different datasets obtained at the same time.  To achieve this
    # the lightcurves need to use the same time bins, and the range of this is
    # set from the overall range of timestamps included in the photometry.
    # This is estimated from a central subset of stars, since this is otherwise
    # unnecessarily computationally intensive.
    smin = int(quad_phot.shape[0]/2)
    smax = int(smin + 100)
    ts = (quad_phot[smin:smax,:,0]).flatten()
    idx = np.where(ts > 2450000.0)
    survey_start = float(int(ts[idx].min()))
    survey_end = float(int(ts[idx].max()))+1.0
    bin_width = 1.0
    log.info('Survey start='+str(survey_start)+' and end='+str(survey_end))
    survey_time_bins = np.arange(survey_start+(bin_width/2.0),
                                 survey_end+(bin_width/2.0), bin_width)

    log.info('Starting normalization for star '+str(field_idx))
    (xmatch, quad_phot) = normalize_star_datasets(params, xmatch, quad_phot,
                                survey_time_bins, field_idx, filter_list,
                                log=None, diagnostics=False)
    log.info('Completed normalization')

    # Store the stellar normalization coefficients per dataset to the
    # crossmatch table:
    xmatch.save(params['crossmatch_file'])

    # Output the updated photometry
    normalize_photometry.output_quadrant_photometry(params, setup, qid,
                                                    quad_phot, log)

    logs.close_log(log)

def normalize_star_datasets(params, xmatch, quad_phot, survey_time_bins,
                            field_idx, filter_list,
                            log=None, diagnostics=False):
    """Function bins the star's lightcurve in time, and calculates the weighted
    mean magnitude offset of each dataset from the primary reference lightcurve
    in each filter, from the binned data, excluding outliers. """

    # Extract the lightcurve of the given star in its constituent datasets:
    lc = load_norm_field_photometry_for_star_idx(params, field_idx,
                                                xmatch, quad_phot,log=None)

    for f in filter_list:
        # Fetch the primary reference lightcurve for this star
        #pri_ref_code = xmatch.get_dataset_shortcode(xmatch.reference_datasets[f])
        pri_ref_code = xmatch.reference_datasets[f]
        pri_ref_lc = lc[pri_ref_code]

        # The primary reference lightcurve for this filter is used to determine
        # the time bins that will be used for the dataset lightcurves
        binned_pri_ref_lc = bin_lc_in_time(pri_ref_lc, survey_time_bins)

        if log: log.info('Binned primary reference lightcurve '+pri_ref_code)

        for dset in lc.keys():
            data = lc[dset]
            if f in dset:
                if dset != pri_ref_code:
                    # Bin the dataset's lightcurve using the same time bins as
                    # the primary reference:
                    binned_lc = bin_lc_in_time(data, survey_time_bins)
                    if log: log.info('Binned dataset lightcurve '+dset)

                    # Determine the offset between this dataset and the primary_ref
                    plot_file = path.join(params['red_dir'],
                                'star_norm_residuals_'+dset+'.png')
                    (offset, offset_error) = measure_dataset_offset(binned_pri_ref_lc,
                                                                    binned_lc, log=None,
                                                                    plot_file=plot_file)

                    if log: log.info(dset+' measured offset: '+str(offset)+'+/-'+str(offset_error))

                    # Apply the offset to correct the unbinned lightcurve from the
                    # dataset:
                    if diagnostics: data_orig = copy.deepcopy(data)
                    data = apply_dataset_offset(data, offset, offset_error, log=None)

                    # Store the results
                    lc[dset] = data
                    xmatch = update_mag_offsets_table(xmatch, field_idx,
                                                      dset, offset, offset_error)

                    if diagnostics:
                        fig = plt.figure(1,(10,10))
                        plt.rcParams.update({'font.size': 18})
                        plt.plot(pri_ref_lc[:,0]-2450000.0, pri_ref_lc[:,1], 'mo', label='primary-ref')
                        plt.errorbar(data_orig[:,0]-2450000.0, data_orig[:,1], yerr=data_orig[:,2],
                                mfc='purple', mec='purple', ecolor='purple', marker='x', markersize=2, ls='none',
                                label='Original dataset')
                        plt.errorbar(data[:,0]-2450000.0, data[:,1], yerr=data[:,2],
                                mfc='green', mec='green', ecolor='green', marker='d', markersize=2, ls='none',
                                label='Normalized dataset')
                        plt.xlabel('HJD-2450000.0')
                        plt.ylabel('Mag')
                        plt.legend()
                        plt.savefig(path.join(params['red_dir'],
                                    'star_norm_residual_lc_'+f+'_'+dset+'.png'))
                        plt.close(1)

                        fig = plt.figure(2,(10,10))
                        plt.rcParams.update({'font.size': 18})
                        plt.plot(pri_ref_lc[:,0]-2450000.0, pri_ref_lc[:,1], 'mo', label='primary-ref')
                        plt.plot(binned_pri_ref_lc[:,0]-2450000.0, binned_pri_ref_lc[:,1], 'mo')
                        plt.plot(binned_lc[:,0]-2450000.0, binned_lc[:,1], 'gd')
                        plt.xlabel('HJD-2450000.0')
                        plt.ylabel('Mag')
                        plt.legend()
                        plt.savefig(path.join(params['red_dir'],
                                    'star_norm_binned_lc_'+f+'_'+dset+'.png'))
                        plt.close(2)

    # Store re-normalized photometry in the main quadrant photometry array:
    quad_phot = update_norm_field_photometry_for_star_idx(field_idx,
                                                        xmatch, quad_phot, lc,
                                                        log=None)

    return xmatch, quad_phot

def update_mag_offsets_table(xmatch, field_idx, dset,
                                offset, offset_error):
    cname1 = 'delta_mag_'+xmatch.get_dataset_shortcode(dset)
    cname2 = 'delta_mag_error_'+xmatch.get_dataset_shortcode(dset)
    xmatch.normalizations[cname1][field_idx] = offset
    xmatch.normalizations[cname2][field_idx] = offset_error

    return xmatch

def load_norm_field_photometry_for_star_idx(params, field_idx, xmatch, quad_phot,
                                            log=None):
    """Function to load the photometry of a single star from the corresponding
    quadrant's field photometry array, and split the photometry into its
    component datasets.
    Note that this function differs from similar functions in field_lightcurves
    because it performs no filtering of missing measurements, and also uses
    the longhand version of the dataset codes.  This is necessary, in order
    to be able to update the data in the quadrant photometry array later.
    """

    quad_idx = xmatch.field_index['quadrant_id'][field_idx] - 1
    if log:
        log.info('Loading photometry for star field_idx='+str(field_idx)
                +', quad_idx='+str(quad_idx))
    lc = {}
    (mag_col, merr_col) = field_photometry.get_field_photometry_columns('normalized')

    for dataset in xmatch.datasets:
        # Extract the photometry of this object for the images from this dataset,
        # if the field index indicates that the object was measured in this dataset
        if xmatch.field_index[dataset['dataset_code']+'_index'][field_idx] != 0:
            longcode = dataset['dataset_code']

            # Select those images from the HDF5 pertaining to this dataset,
            # then select valid measurements for this star
            idx = np.where(xmatch.images['dataset_code'] == dataset['dataset_code'])[0]

            # Store the photometry
            if len(idx) > 0:
                photometry = np.zeros((len(idx),3))
                photometry[:,0] = quad_phot[quad_idx,idx,0]
                photometry[:,1] = quad_phot[quad_idx,idx,mag_col]
                photometry[:,2] = quad_phot[quad_idx,idx,merr_col]
                lc[longcode] = photometry
                if log: log.info('-> Loading '+str(len(idx))
                        +' datapoints from dataset '+dataset['dataset_code'])
            else:
                if log: log.info('-> No data available from dataset '
                            +dataset['dataset_code'])
    return lc

def update_norm_field_photometry_for_star_idx(field_idx, xmatch,
                                                quad_phot, lc, log=None):
    """Function to update the normalized photometry columns for a single star in
    the corresponding quadrant's field photometry array, based on updated
    lightcurves split into component datasets.
    """

    quad_idx = xmatch.field_index['quadrant_id'][field_idx] - 1
    if log:
        log.info('Updating photometry for star field_idx='+str(field_idx)
                +', quad_idx='+str(quad_idx))

    (mag_col, merr_col) = field_photometry.get_field_photometry_columns('normalized')

    for dset in lc.keys():     # Longcode dataset name format
        data = lc[dset]

        # Select those images from the HDF5 pertaining to this dataset,
        # then select valid measurements for this star
        idx = np.where(xmatch.images['dataset_code'] == dset)[0]
        if log:
            log.info('-> Loading '+str(len(idx))+' datapoints from dataset '+dset
                    +', '+str(len(data))+' datapoints in lightcurve array')

        # Store the photometry
        if len(idx) > 0:
            quad_phot[quad_idx,idx,mag_col] = data[:,1]
            quad_phot[quad_idx,idx,merr_col] = data[:,2]

    return quad_phot


def apply_dataset_offset(lc_data, offset, offset_error, log=None):

    if offset != 0.0 and offset_error != 0.0:
        # Apply the correction only to valid measurements, since these are
        # otherwise filtered out by other parts of the pipeline
        idx1 = np.where(lc_data[:,1] > 0.0)[0]
        idx2 = np.where(lc_data[:,2] > 0.0)[0]
        idx = list(set(idx1).intersection(set(idx2)))

        lc_data[idx,1] += offset
        lc_data[idx,2] = np.sqrt( lc_data[idx,2]*lc_data[idx,2]
                                + offset_error*offset_error )
        if log: log.info('-> Applied calculated offset to original dataset lightcurve')
    else:
        if log: log.info('-> Invalid offset, no change made to original lightcurve')

    return lc_data

def measure_dataset_offset(binned_pri_ref_lc, binned_lc, log=None, plot_file=None):
    """Function to determine the magnitude offset of a dataset's lightcurve
    from the primary reference lightcurve in the same filter.

    This function takes two lightcurves binned using the same time bins, but
    doesn't assume that both lightcurves have measurements in each bin.

    The function returns the magnitude offset such that:
    offset = primary_ref - dataset

    So that:
    dataset + offset -> primary_ref
    """

    # Parameters required for the RANSAC fit:
    min_samples = 2
    residuals_threshold = 0.1

    # Identify those bins where both lightcurves have valid measurements
    select = np.where(np.logical_and(np.isfinite(binned_pri_ref_lc[:,1]),
                            np.isfinite(binned_lc[:,1])))[0]
    if log: log.info('Identified '+str(len(select))
                +' bins with contemporaneous lightcurve measurements')

    if len(select) >= min_samples:

        # Calculate photometric residuals between the bins, and construct the
        # data array for RANSAC:
        residuals = np.zeros((len(select),3))
        residuals[:,0] = binned_pri_ref_lc[select,0]
        residuals[:,1] = binned_pri_ref_lc[select,1] - binned_lc[select,1]
        residuals[:,2] = np.sqrt((binned_pri_ref_lc[select,1]*binned_pri_ref_lc[select,1]) +
                                (binned_lc[select,1]*binned_lc[select,1]))

        # Identify in/outliers, and calculate the transformation between the two:
        (transform, inliers) = ransac(residuals, LineModelND,
                                        min_samples=min_samples,
                                        residual_threshold=residuals_threshold)
        offset = transform.params[0][1]

        # Subtracting the measured offset from the residuals, calculate the
        # weighted mean of the remaining scatter as an uncertainty on the offset
        jdx1 = np.where(residuals[:,2] > 0.0)[0]
        jdx2 = np.where(abs(residuals[:,1]-offset) <= residuals_threshold)[0]
        jdx = list(set(jdx1).intersection(set(jdx2)))
        res = (residuals[jdx,1] - offset)
        err_squared_inv = 1.0 / (residuals[jdx,2]*residuals[jdx,2])
        wmean =  (res * err_squared_inv).sum() / (err_squared_inv.sum())

        if plot_file:
            fig = plt.figure(1,(10,10))
            plt.rcParams.update({'font.size': 18})
            plt.plot(residuals[:,0]-2450000.0, residuals[:,1], 'mo')
            plt.plot([(residuals[:,0]-2450000.0).min(), (residuals[:,0]-2450000.0).max()],
                        [offset, offset], 'g-')
            plt.plot([(residuals[:,0]-2450000.0).min(), (residuals[:,0]-2450000.0).max()],
                        [offset-wmean, offset-wmean], 'g--')
            plt.plot([(residuals[:,0]-2450000.0).min(), (residuals[:,0]-2450000.0).max()],
                        [offset+wmean, offset+wmean], 'g--')
            plt.xlabel('HJD-2450000.0')
            plt.ylabel('Mag')
            plt.savefig(plot_file)
            plt.close(1)

    # If the number of bins with valid measurements in both lightcurves is
    # too small to reliably measure an offset, return zero values:
    else:
        offset = 0.0
        wmean = 0.0

    return offset, wmean

def bin_lc_in_time(lc, time_bins):
    """Function to bin a lightcurve array into time bins, by default 24hrs
    wide.  Function expects a lightcurve in the form of a numpy array with
    columns HJD, mag, mag_error, qc_flag."""

    # Index all datapoints, assigning them to one of the time bins
    time_index = np.digitize(lc[:,0], time_bins)

    idx1 = np.where(lc[:,1] > 0.0)[0]
    idx2 = np.where(lc[:,2] > 0.0)[0]
    valid = set(idx1).intersection(set(idx2))

    binned_lc = np.zeros((len(time_bins),3))
    binned_lc[:,0] = time_bins
    for b in range(0,len(time_bins),1):
        idx1 = np.where(time_index == b)[0]
        idx = list(set(idx1).intersection(valid))
        if len(idx) > 0:
            (wmean, werror) = calc_weighted_mean_datapoint(lc[idx,1], lc[idx,2])
            binned_lc[b,1] = wmean
            binned_lc[b,2] = werror
        else:
            binned_lc[b,1] = np.NaN
            binned_lc[b,2] = np.NaN

    return binned_lc

def calc_weighted_mean_datapoint(data, errs):
    """Expects input array of the form: columns HJD, mag, mag_error, qc_flag"""

    err_squared_inv = 1.0 / (errs*errs)
    wmean =  (data * err_squared_inv).sum() / (err_squared_inv.sum())
    werror = np.sqrt( 1.0 / (err_squared_inv.sum()) )

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
    params['phot_type'] = 'normalized'

    return setup, params


if __name__ == '__main__':
    (setup, params) = get_args()
    run_star_normalization(setup, **params)
