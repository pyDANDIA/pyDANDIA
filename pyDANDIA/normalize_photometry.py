from os import path, rename
from sys import argv
import numpy as np
from pyDANDIA import crossmatch
from pyDANDIA import hd5_utils
from pyDANDIA import logs
from pyDANDIA import pipeline_setup
from pyDANDIA import plot_rms

def run_phot_normalization(setup, **params):
    """Function to normalize the photometry between different datasets taken
    of the same field in the same filter but with different instruments.

    Since different reference images are used for different datasets, there
    remain small offsets in the calibrated magnitudes of the lightcurves.
    This function compares the mean magnitudes of constant stars in the field
    to calculate these small magnitude offsets and correct for them.
    """

    log = logs.start_stage_log( setup.red_dir, 'postproc_phot_norm' )

    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(params['crossmatch_file'],log=log)

    # Using corrected_mag columns:
    mag_col = 5
    merr_col = 6

    # Identify the datasets to be used as the primary reference in each
    # filter:
    xmatch.id_primary_datasets_per_filter()
    print(xmatch.reference_datasets)

    # Add columns to the dataset Table to hold the photometric calibration
    # parameters
    (ndset, ncol) = xmatch.datasets.shape
    xmatch.datasets.add_column(np.zeros(ndset), name='norm_a0', index=ncol+1)
    xmatch.datasets.add_column(np.zeros(ndset), name='norm_a1', index=ncol+2)
    xmatch.datasets.add_column(np.zeros(ndset), name='norm_covar_0', index=ncol+3)
    xmatch.datasets.add_column(np.zeros(ndset), name='norm_covar_1', index=ncol+4)
    xmatch.datasets.add_column(np.zeros(ndset), name='norm_covar_2', index=ncol+5)
    xmatch.datasets.add_column(np.zeros(ndset), name='norm_covar_3', index=ncol+6)

    # Extract list of filters from xmatch.images['filter'] column
    filter_list = np.unique(xmatch.images['filter'].data)

    # Loop over all quadrants.  XXX CURRENTLY LIMITED FOR TESTING XXX
    # THIS NEEDS TO COMBINE PHOTOMETRY FROM QUADS TO AVOID BOUNDARY ISSUES
    read_phot = False
    if read_phot:
        for quad in range(1,2,1):
            phot_file = path.join(setup.red_dir,params['field_name']+'_quad'+str(quad)+'_photometry.hdf5')
            phot_data = hd5_utils.read_phot_from_hd5_file(phot_file, return_type='array')

            # Loop over all filters
            for filter in filter_list:

                # Extract lightcurves for all stars in quadrant for a given filter,
                # combining data from multiple cameras.
                image_index = np.where(xmatch.images['filter'] == filter)[0]
                print(image_index)
                phot_data_filter = phot_data[:,image_index,:]
                print(phot_data_filter.shape)

                # Plot a multi-site initial RMS diagram for reference
                phot_statistics = np.zeros( (len(phot_data_filter),4) )
                (wmean,werror) = plot_rms.calc_weighted_mean_2D(phot_data_filter, mag_col, merr_col)

                tmp = plot_rms.calc_weighted_rms(phot_data_filter, phot_statistics[:,0], mag_col, merr_col)

                plot_rms.plot_rms(phot_statistics, params, log)
                rename(path.join(params['red_dir'],'rms.png'),
                       path.join(params['red_dir'],'rms_prenorm'+str(filter)+'.png'))

            # Append the data from this quadrant to complete the dataset

    # Normalize the photometry of each dataset to that of the reference
    # image in the primary reference dataset in that filter
    for filter in filter_list:
        # Extract the reference image photometry for the primary-ref dataset
        # for this filter
        ref_datacode = xmatch.reference_datasets[filter]
        ref_phot = np.zeros((len(xmatch.stars),2))
        ref_phot[:,0] = xmatch.stars['cal_'+filter+'_mag_'+ref_datacode]
        ref_phot[:,1] = xmatch.stars['cal_'+filter+'_magerr_'+ref_datacode]

        # Identify constant stars based on RMS
        (mean_mag, mean_magerr) = calc_weighted_mean(ref_phot)
        rms = calc_weighted_rms(ref_phot, mean_mag)
        constant_stars = id_constant_stars(data, rms)

        # Extract the lightcurves for all other datasets in turn
        dataset_index = np.where(self.datasets['filter'] == filter)[0]

        for idset in dataset_index:
            dset_datacode = self.datasets['dataset_code'][idset]
            ref_phot = np.zeros((len(xmatch.stars),2))
            dset_phot[:,0] = xmatch.stars['cal_'+filter+'_mag_'+dset_datacode]
            dset_phot[:,1] = xmatch.stars['cal_'+filter+'_magerr_'+dset_datacode]

            # Calculate their weighted offset relative to the primary-ref
            # dataset for the filter
            (dmag, dmag_err) = calc_phot_offset(ref_phot, dset_phot, constant_stars)
            xmatch.datasets[idset]['norm_coeff_a0']
    # Plot delta mag histogram, delta mag vs mag, delta mag vs position

    # Compute corrected mag

    # Plot a multi-site final RMS diagram for comparison

    # Output updated phot.hdf file

    logs.close_log(log)

    status = 'OK'
    report = 'Completed successfully'
    return status, report

def calc_phot_normalization(ref_phot, dset_phot, constant_stars):
    """Function to calculate the magnitude offset between the magnitudes of
    constant stars in the given dataset relative to the primary reference.
    """
    fit = np.array([1,0])
    covar_fit = np.zeros((3,3))
    (fit,covar_fit) = calibrate_photometry.calc_transform(fit,
                                                    ref_phot[constant_stars,0],
                                                    dset_phot[constant_stars,0])
    XXX Store in xmatch.datasets

    return dmag, dmag_err

def id_constant_stars(data, rms):
    """Function identifies constant stars as those with an RMS in the lowest
    10 - 25% of the set.  This excludes both stars with high scatter and those
    with artificially low scatter due to having few measurements."""

    rms_range = rms.max() - rms.min()
    min_cut = rms.min() + (rms_range)*0.1
    max_cut = rms.min() + (rms_range)*0.25

    constant_stars = np.where((rms >= min_cut) && (rms <= max_cut))[0]
    return constant_stars

def calc_weighted_mean(data):

    mask = np.invert(np.logical_and(data[:,0] > 0.0, data[:,1] > 0.0))
    mags = np.ma.array(data[:,0], mask=mask)
    errs = np.ma.array(data[:,1], mask=mask)

    idx = np.where(mags > 0.0)
    err_squared_inv = 1.0 / (errs*errs)
    wmean =  (mags * err_squared_inv).sum(axis=1) / (err_squared_inv.sum(axis=1))
    werror = np.sqrt( 1.0 / (err_squared_inv.sum(axis=1)) )

    return wmean, werror

def calc_weighted_rms(data, mean_mag):

    mask = np.invert(np.logical_and(data[:,0] > 0.0, data[:,1] > 0.0))
    mags = np.ma.array(data[:,0], mask=mask)
    errs = np.ma.array(data[:,1], mask=mask)

    err_squared_inv = 1.0 / (errs*errs)
    dmags = (mags.transpose() - mean_mag).transpose()
    rms =  np.sqrt( (dmags**2 * err_squared_inv).sum(axis=1) / (err_squared_inv.sum(axis=1)) )

    return rms

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
    (status, report) = run_phot_normalization(setup, **params)
