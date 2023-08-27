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
from pyDANDIA import create_star_normalization_tables
from astropy.table import Table, Column
import matplotlib.pyplot as plt
from skimage.measure import ransac, LineModelND
import copy

VERSION = 'star_norm_v1.2'

def run_star_normalization(setup, **params):

    log = logs.start_stage_log( setup.red_dir, 'postproc_star_norm', version=VERSION )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(params['crossmatch_file'],log=log)
    xmatch.id_primary_datasets_per_filter()

    image_sets = xmatch.get_imagesets()
    filter_list = np.unique(xmatch.images['filter'].data)
    reference_datasets = define_reference_datasets(xmatch, filter_list)

    for qid in range(1,5,1):

        log.info('Loading timeseries photometry for quadrant '+str(qid)+'...')
        file_path = path.join(setup.red_dir, params['field_name']\
                        +'_quad'+str(qid)+'_photometry.hdf5')
        quad_phot = hd5_utils.read_phot_from_hd5_file(file_path, return_type='array')
        log.info('...completed')

        # Mask the main photometry array, to avoid using invalid data in the calculations
        log.info('Masking invalid photometry in the data array')
        (mag_col, mag_err_col) = field_photometry.get_field_photometry_columns('normalized')
        quad_phot = hd5_utils.mask_phot_array(quad_phot, mag_col, mag_err_col)
        log.info('...completed')

        # The normalization process compares the photometry measurements of a given
        # star from different datasets obtained at the same time.
        survey_time_bins = calc_survey_time_bins(quad_phot, log)
        log.info('Survey time bins: '+repr(survey_time_bins))

        # Assign the images from each dataset to the survey_time_bins based on
        # their HJDs.
        survey_time_index = np.digitize(xmatch.images['hjd'], survey_time_bins)
        if log: log.info('Digitized the timestamps for all images')

        # Loop over the set of datasets to use as the primary reference.
        # Each pri_ref_datasets in the list is a dictionary of the
        # datasets per filter
        for pri_ref_datasets in reference_datasets:
            log.info('Normalizing stars relative to primary reference datasets: '
                        +repr(pri_ref_datasets))

            # Bin the photometry for all stars from primary reference datasets
            # for all filters
            binned_ref_phot = {}
            for f, longcode in pri_ref_datasets.items():
                bin_data = bin_photometry_datasets(xmatch, quad_phot,
                                                    survey_time_bins,
                                                    survey_time_index,
                                                    longcode,
                                                    mag_col, mag_err_col,
                                                    log=log)
                binned_ref_phot[longcode] = bin_data[longcode]
            print('1: binned_ref_phot ',binned_ref_phot)

            # Loop over all datasets - note that this means each filter
            # is handled separately
            for dataset in xmatch.datasets:
                longcode = dataset['dataset_code']

                # Bin the photometry for all stars from the current dataset
                bin_data = bin_photometry_datasets(xmatch, quad_phot,
                                                        survey_time_bins,
                                                        survey_time_index,
                                                        longcode,
                                                        mag_col, mag_err_col,
                                                        log=log)
                print('2: bin_data ',bin_data)

                # Combined the dataset's binned photometry with that from
                # the primary reference datasets in all filters
                binned_photometry['longcode'] = bin_data[longcode]
                for dcode, dphot in binned_ref_phot.items():
                    binned_photometry[dcode] = dphot
                print('3: binned_photometry ',binned_photometry)

                # Normalize the photometry for all stars
                (xmatch, quad_phot) = normalize_star_datasets(params,
                                        xmatch, quad_phot, qid,
                                        binned_photometry, filter_list,
                                        pri_ref_datasets,
                                        reference_datasets,
                                        mag_col, mag_err_col,
                                        log=log)

        # Output the updated photometry
        quad_phot = hd5_utils.unmask_phot_array(quad_phot)
        normalize_photometry.output_quadrant_photometry(params, setup, qid,
                                                        quad_phot, log)

    # Store the stellar normalization coefficients per dataset to the
    # crossmatch table:
    xmatch.save(params['crossmatch_file'])
    hd5_utils.write_normalizations_hd5(setup.red_dir, params['field_name'],
                                        normalizations)

    logs.close_log(log)

def define_reference_datasets(xmatch, filter_list):
    """
    Function to define the datasets to be used as the reference to normalize
    the photometry to.
    Overall the primary reference datasets (defined to be LSC-DOMA) provide
    measurements for ~60% of detected stars, but due to variations in the
    telescope pointings, they do not provide photometry for the remaining stars.
    Additional stars can be normalized by considering other datasets as
    the 'primary reference', if data from LSC-DOMA is not available.
    In these cases, data from CPT-DOMA and COJ-DOMA will be considered to be
    the reference in that order of preference.
    Stars that are not measured in any DOMA dataset (~12%) are NOT normalized,
    since the remaining datasets are highly concentrated in time due to the REA
    strategy, and therefore do not provide sufficent overlapping time bins.
    """

    if not hasattr(xmatch, 'ref_preference_order'):
        xmatch.id_primary_datasets_per_filter()

    reference_datasets = []

    for primary in xmatch.ref_preference_order:
        ref_datasets = {}
        for f in filter_list:
            for dset_code in xmatch.datasets['dataset_code']:
                if primary in dset_code and f in dset_code:
                    ref_datasets[f] = dset_code
        reference_datasets.append(ref_datasets)

    return reference_datasets

def bin_photometry_datasets(xmatch, quad_phot, survey_time_bins, survey_time_index,
                            longcode, mag_col, mag_err_col, log=None):
    """Each dataset consists of a set of images with given timestamps, that
    are recorded in the crossmatch table.  Since these are essentially the same
    for all stars (modulo instances where an individual star isn't measured for
    some reason) we save compute time by assigning all of the images of each
    dataset to a survey_time_bin in advance.
    longcode = dataset['dataset_code']
    """

    if log: log.info('Starting to bin the photometry for each dataset')
    binned_phot = {}

    if log: log.info('-> Binning data for '+longcode)

    binned_data = np.zeros((quad_phot.shape[0],len(survey_time_bins),3))

    # Extract the image array indices for the images from this dataset,
    # and their corresponding timestamps
    idx1 = set(np.where(xmatch.images['dataset_code'] == longcode)[0])

    for b in range(0,len(survey_time_bins),1):
        binned_data[:,b,0].fill(survey_time_bins[b])

        # Identify whether this dataset has any measurements in this bin
        idx2 = np.where(survey_time_index == b)[0]
        idx = list(idx1.intersection(set(idx2)))
        if len(idx) > 0:
            # For all stars, calculate the weighted mean magnitude combining
            # all photometric measurements in this bin
            # wmean, werror -> arrays of length nstars
            (wmean, werror) = plot_rms.calc_weighted_mean_2D(quad_phot[:,idx,:],
                                                             mag_col, mag_err_col)
            binned_data[:,b,1] = wmean
            binned_data[:,b,2] = werror
        else:
            binned_data[:,b,1].fill(np.NaN)
            binned_data[:,b,2].fill(np.NaN)

    binned_phot[longcode] = binned_data

    return binned_phot

def calc_survey_time_bins(quad_phot, log):
    """The normalization process compares the photometry measurements of a given
    star from different datasets obtained at the same time.  To achieve this
    the lightcurves need to use the same time bins, and the range of this is
    set from the overall range of timestamps included in the photometry.
    This is estimated from a central subset of stars, since this is otherwise
    unnecessarily computationally intensive."""

    # Choose a fairly central subset of star IDs
    smin = int(quad_phot.shape[0]/2)
    smax = int(smin + 100)

    # Select datapoints with valid measurements:
    ts = (quad_phot[smin:smax,:,0]).flatten()
    idx = np.where(ts > 2450000.0)

    # Generate a set of bins between the minimum and maximum HJDs:
    survey_start = float(int(ts[idx].min()))
    survey_end = float(int(ts[idx].max()))+1.0
    bin_width = 1.0

    log.info('Survey start='+str(survey_start)+' and end='+str(survey_end))
    survey_time_bins = np.arange(survey_start+(bin_width/2.0),
                                 survey_end+(bin_width/2.0), bin_width)

    return survey_time_bins

def normalize_star_datasets(params, xmatch, quad_phot, qid, binned_phot,
                            filter_list, pri_ref_datasets, reference_datasets,
                            mag_col, mag_err_col,
                            log=None, verbose=False):
    """Function bins the star's lightcurve in time, and calculates the weighted
    mean magnitude offset of each dataset from the primary reference lightcurve
    in each filter, from the binned data, excluding outliers. """

    # Establish the path to the HDF5 file storing the magnitude offsets between
    # each of the primary reference datasets and all other datasets
    normalizations_file = path.join(params['red_dir'],
                        params['field_name']+'_star_dataset_normalizations.hdf5')

    for f in filter_list:
        # Identify the code of the dataset to be used as the primary reference
        pri_ref_code = pri_ref_datasets[f]
        log.info('Using '+pri_ref_code+' as the primary reference in filter '+f)

        # Load the table of normalization coefficients for this primary reference
        # dataset:
        column_names = create_star_normalization_tables.get_table_columns(xmatch)
        short_pri_ref_code = xmatch.get_dataset_shortcode(pri_ref_code)
        short_pri_ref_code = short_pri_ref_code.split('_')[0]
        normalizations = hd5_utils.read_normalizations_table(normalizations_file,
                                                             short_pri_ref_code,
                                                             column_names)

        # The primary reference lightcurve for this filter is used to determine
        # the time bins that will be used for the dataset lightcurves
        binned_pri_ref_data = binned_phot[pri_ref_code]

        for dset in binned_phot.keys():
            if f in dset:
                if dset != pri_ref_code:
                    if log: log.info('Measuring magnitude offsets for '+dset)
                    binned_data = binned_phot[dset]

                    # Determine the offset between this dataset and the primary_ref
                    # for all stars in all time bins:
                    (residuals,status) = calc_residuals_between_datasets(binned_pri_ref_data,
                                                                binned_data)

                    # Since this process uses RANSAC to optimize
                    # the in/outliers independently for each star in this quadrant,
                    # this cannot be an array operation
                    quad_offsets = np.zeros((binned_data.shape[0],2))
                    select_stars = get_dataset_stars_in_quadrant(xmatch, qid,
                                                        reference_datasets,
                                                        pri_ref_code, dset, f)

                    if status:
                        if len(select_stars) > 0:
                            if log: log.info('Starting RANSAC optimization for '
                                        +str(len(select_stars))+' stars')
                            for j in range(0,len(select_stars),1):
                                fidx = select_stars['field_id'][j] - 1
                                quad_idx = select_stars['quadrant_id'][j] - 1
                                #plot_file = path.join(params['red_dir'],'ransac',
                                #            'residuals_'+dset+'_'+str(fidx+1)+'.png')
                                (dm, dmerr) = measure_dataset_offset(residuals[quad_idx,:,:],
                                                                    log=None, plot_file=None)
                                quad_offsets[quad_idx,0] = dm
                                quad_offsets[quad_idx,1] = dmerr
                                if log and verbose:
                                    log.info('-> field ID: '+str(fidx+1)+', quadrant '
                                                +str(qid)+' quadrant ID: '+str(quad_idx+1)
                                                 +' offset='+str(dm)+' +/- '+str(dmerr))
                            if log:
                                log.info('Completed RANSAC optimizations')

                            # Apply these offsets to correct the unbinned lightcurve
                            # for all stars in the quadrant for this dataset:
                            quad_phot = apply_dataset_offsets(xmatch, quad_phot, dset,
                                                              quad_offsets,
                                                              mag_col, mag_err_col,
                                                              log=None)
                            if log: log.info('Applied stellar photometric normalizations to main data array')

                        else:
                            if log:
                                log.info('No stars in quadrant '+str(qid)
                                    +' to normalize for this combination of primary reference '+pri_ref_code
                                    +' and dataset '+dset)
                    else:
                        if log: log.info('No valid residuals between this dataset and the primary reference')

                    # Store the coefficients
                    normalizations = update_mag_offsets_table(xmatch, qid, pri_ref_code,
                                                              dset, select_stars, quad_offsets,
                                                              normalizations)
                    if log: log.info('Updated the array with normalization coefficients')

        hd5_utils.update_normalizations_table(normalizations_file,
                                              short_pri_ref_code,
                                              normalizations)
        if log: log.info('Stored the updated normalization coefficients to file')

    return xmatch, quad_phot

def get_dataset_stars_in_quadrant(xmatch, qid, reference_datasets, ref_dset, dset, filter):
    """Function returns an array of the field indices (not IDs) of stars in
    a given quadrant which were identified in both primary reference and
    working datasets"""

    if xmatch.ref_preference_order[0] in ref_dset:
        stars = range(0,len(xmatch.field_index),1)
        idx4 = set(stars)
    elif xmatch.ref_preference_order[1] in ref_dset:
        priref = reference_datasets[0][filter]
        idx4 = np.where(xmatch.field_index[priref+'_index'] == 0)[0]
    elif xmatch.ref_preference_order[2] in ref_dset:
        priref = reference_datasets[0][filter]
        idx4a = np.where(xmatch.field_index[priref+'_index'] == 0)[0]
        priref2 = reference_datasets[1][filter]
        idx4b = np.where(xmatch.field_index[priref2+'_index'] == 0)[0]
        idx4 = set(idx4a).intersection(set(idx4b))

    idx1 = np.where(xmatch.field_index['quadrant'] == qid)[0]
    idx2 = np.where(xmatch.field_index[ref_dset+'_index'] > 0)[0]
    idx3 = np.where(xmatch.field_index[dset+'_index'] > 0)[0]
    idx = set(idx1).intersection(set(idx2))
    idx = idx.intersection(set(idx3))
    idx = list(idx.intersection(set(idx4)))

    col_list = [ Column(name='field_id', data=xmatch.field_index['field_id'][idx]),
                 Column(name='quadrant_id', data=xmatch.field_index['quadrant_id'][idx])]
    select_stars = Table(col_list)

    return select_stars

def calc_residuals_between_datasets(binned_data1, binned_data2):

    # Mask out residuals datapoints if the measurements are invalid in both
    # binned datasets
    mask1 = np.logical_and(np.isfinite(binned_data1), binned_data1 > 0.0)
    mask2 = np.logical_and(np.isfinite(binned_data2), binned_data2 > 0.0)
    mask = np.invert(np.logical_and(mask1, mask2))

    # Calculate the residuals
    data = np.zeros((binned_data1.shape[0],binned_data1.shape[1],3))
    data[:,:,0] = binned_data1[:,:,0]
    data[:,:,1] = binned_data1[:,:,1] - binned_data2[:,:,1]
    data[:,:,2] = np.sqrt((binned_data1[:,:,2]*binned_data1[:,:,2]) +
                            (binned_data2[:,:,2]*binned_data2[:,:,2]))

    # Test whether there are any valid residuals:
    valid = np.where(mask[:,:,1] == False)[0]
    if len(valid) > 0:
        test = True
    else:
        test = False

    return np.ma.masked_array(data, mask=mask), test

def update_mag_offsets_table(xmatch, qid, pri_ref_code, dset, select_stars,
                             quad_offsets, normalizations):

    # Get the field indices of this quadrant's stars
    quad_idxs = select_stars['quadrant_id'] - 1
    field_idxs = select_stars['field_id'] - 1
    cname1 = 'delta_mag_'+xmatch.get_dataset_shortcode(dset)
    cname2 = 'delta_mag_error_'+xmatch.get_dataset_shortcode(dset)

    normalizations[cname1][field_idxs] = quad_offsets[quad_idxs,0]
    normalizations[cname2][field_idxs] = quad_offsets[quad_idxs,1]

    return normalizations

def apply_dataset_offsets(xmatch, quad_phot, dset,
                          quad_offsets, mag_col, mag_err_col, log=None):

    # Select the images corresponding to this dataset:
    idx = np.where(xmatch.images['dataset_code'] == dset)[0]

    # Apply the correction only to valid measurements, since these are
    # otherwise filtered out by other parts of the pipeline
    if len(idx) > 0:
        for i in idx:
            quad_phot[:,i,mag_col] += quad_offsets[:,0]
            quad_phot[:,i,mag_err_col] = np.sqrt( quad_phot[:,i,mag_err_col]*quad_phot[:,i,mag_err_col]
                                                + quad_offsets[:,1]*quad_offsets[:,1] )

    return quad_phot

def measure_dataset_offset(residuals, log=None, plot_file=None):
    """Function to determine the magnitude offset of a dataset's lightcurve
    from the primary reference lightcurve in the same filter.

    This function takes two lightcurves binned using the same time bins, but
    doesn't assume that both lightcurves have measurements in each bin.

    The function returns the magnitude offset such that:
    offset = primary_ref - dataset

    So that:
    dataset + offset -> primary_ref

    Note: Expects a masked array
    """

    # Parameters required for the RANSAC fit:
    min_samples = 2
    residuals_threshold = 0.1

    # Identify those bins where both lightcurves have valid measurements
    mask = np.ma.getmask(residuals)
    select = np.where(mask[:,1] == False)[0]
    if log: log.info('Identified '+str(len(select))
                +' bins with contemporaneous lightcurve measurements')

    if len(select) > min_samples:

        # Identify in/outliers, and calculate the transformation between the two:
        (transform, inliers) = ransac(residuals[select,:].data, LineModelND,
                                        min_samples=min_samples,
                                        residual_threshold=residuals_threshold)
        offset = transform.params[0][1]

        # Subtracting the measured offset from the residuals, calculate the
        # weighted mean of the remaining scatter as an uncertainty on the offset
        jdx1 = np.where(residuals[:,2] > 0.0)[0]
        jdx2 = np.where(abs(residuals[:,1]-offset) <= residuals_threshold)[0]
        jdx = list(set(jdx1).intersection(set(jdx2)))
        res = abs(residuals[jdx,1] - offset)
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
