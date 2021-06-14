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
    qcflag_col = 14

    # Extract list of filters from xmatch.images['filter'] column
    filter_list = np.unique(xmatch.images['filter'].data)

    # Loop over all quadrants.  XXX CURRENTLY LIMITED FOR TESTING XXX
    for quad in range(1,2,1):
        phot_file = path.join(setup.red_dir,params['field_name']+'_quad'+str(quad)+'_photometry.hdf5')
        phot_data = hd5_utils.read_phot_from_hd5_file(phot_file, return_type='array')

        # Loop over all filters
        for filter in filter_list:

            # Extract lightcurves for all stars in quadrant for a given filter,
            # combining data from multiple cameras.
            image_index = np.where(xmatch.images['filter'] == filter)[0]
            phot_data_filter = phot_data[:,image_index,:]
            mask = np.empty(phot_data_filter.shape)
            mask.fill(False)
            idx = np.where(phot_data[:,:,qcflag_col] > 0.0)
            mask[idx] = True
            print(mask)

            # Plot a multi-site initial RMS diagram for reference
            phot_statistics = np.zeros( (len(phot_data_filter),4) )
            (phot_statistics[:,0], phot_statistics[:,3]) = plot_rms.calc_weighted_mean_2D(phot_data_filter, mag_col, merr_col, mask=mask)
            phot_statistics[:,1] = plot_rms.calc_weighted_rms(phot_data_filter, phot_statistics[:,0], mag_col, merr_col, mask=mask)
            phot_statistics[:,2] = plot_rms.calc_percentile_rms(phot_data_filter, phot_statistics[:,0], mag_col, merr_col, mask=mask)
            plot_rms.plot_rms(phot_statistics, params, log)
            rename(path.join(params['red_dir'],'rms.png'),
                   path.join(params['red_dir'],'rms_prenorm_'+str(filter)+'.png'))

            # Extract the lightcurves for the primary-ref dataset for this filter

            # Identify constant stars based on RMS

            # Extract the lightcurves for all other datasets in turn, and
            # calculate their weighted offset relative to the primary-ref
            # dataset for the filter

            # Plot delta mag histogram, delta mag vs mag, delta mag vs position

            # Compute corrected mag

            # Plot a multi-site final RMS diagram for comparison

            # Output updated phot.hdf file

    logs.close_log(log)

    status = 'OK'
    report = 'Completed successfully'
    return status, report

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
