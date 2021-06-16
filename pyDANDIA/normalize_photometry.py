from os import path, rename
from sys import argv
import numpy as np
from pyDANDIA import crossmatch
from pyDANDIA import hd5_utils
from pyDANDIA import logs
from pyDANDIA import pipeline_setup
from pyDANDIA import plot_rms
from pyDANDIA import field_photometry
import matplotlib.pyplot as plt

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
        phot_data = field_photometry.mask_phot_array_by_qcflag(phot_data)

        # Loop over all filters
        for filter in filter_list:

            # Extract lightcurves for all stars in quadrant for a given filter,
            # combining data from multiple cameras.
            search_criteria = {'filter': filter}
            phot_data_filter = field_photometry.extract_photometry_by_image_search(xmatch, phot_data, search_criteria, log=log)

            # Plot a multi-site initial RMS diagram for reference
            phot_statistics = calc_rms(phot_data_filter, mag_col, merr_col)
            plot_rms.plot_rms(phot_statistics, params, log)
            rename(path.join(params['red_dir'],'rms.png'),
                   path.join(params['red_dir'],'rms_prenorm_'+str(filter)+'_quad'+str(quad)+'.png'))

            # Extract the lightcurves for the primary-ref dataset for this filter
            idx = np.where(xmatch.datasets['primary_ref'] == 1)
            primary_ref_code = xmatch.datasets['dataset_code'][idx][0]
            primary_ref_code_filter = '_'.join(primary_ref_code.split('_')[0:2])+'_'+filter
            search_criteria = {'dataset_code': primary_ref_code_filter}
            primary_ref_phot = field_photometry.extract_photometry_by_image_search(xmatch, phot_data, search_criteria, log=log)

            # Identify constant stars based on RMS
            find_constant_stars(params, primary_ref_phot, filter, quad, log, diagnostics=True)

            # Extract the lightcurves for all other datasets in turn, and
            # calculate their weighted magnitude offset relative to the primary-ref
            # dataset for this filter based on the constant star set

            # Plot delta mag histogram, delta mag vs mag, delta mag vs position

            # Compute normalized mags for this dataset and filter

            # Plot a multi-site final RMS diagram for comparison

            # Output updated phot.hdf file

    logs.close_log(log)

    status = 'OK'
    report = 'Completed successfully'
    return status, report

def find_constant_stars(params, phot_data_filter, filter, quad, log, diagnostics=False):

    # Using corrected mag columns
    mag_col = 5
    merr_col = 6
    mag_min = 14.0
    mag_max = 22.0
    niter = 3
    constant_stars = np.arange(0,len(phot_data_filter),1)

    for iter in range(0,niter,1):
        print('N constant stars: '+str(len(constant_stars)))
        phot_statistics = calc_rms(phot_data_filter[constant_stars], mag_col, merr_col)

        # Bin the data by mean magnitude:
        bin_width = 0.5
        bins = np.arange(mag_min, mag_max, bin_width)
        mag_binned = np.zeros(len(bins))
        rms_binned = np.zeros(len(bins))
        rms_std_binned = np.zeros(len(bins))
        mag_idx = np.digitize(phot_statistics[:,0],bins)
        print('Binned data')

        # Calculate the mean mag, RMS and scatter per bin
        new_constant_stars = np.array([], dtype='int')
        for b in range(0,len(bins),1):
            jdx = np.where(mag_idx == b)[0]
            if len(jdx) > 0:
                mag_binned[b] = phot_statistics[jdx,0].mean()
                rms_binned[b] = phot_statistics[jdx,1].mean()
                rms_std_binned[b] = phot_statistics[jdx,1].std()

            # Identify constant stars as those in the lower half of the population
            # per bin, i.e. those with RMS < mean RMS per bin:
            select = np.where(phot_statistics[jdx,0] <= rms_binned[b])[0]
            new_constant_stars = np.concatenate( (new_constant_stars, jdx[select]) )
        print('Reselected constant stars')
        print(new_constant_stars)

        constant_stars = new_constant_stars

        if diagnostics:
            phot_statistics = calc_rms(phot_data_filter[constant_stars], mag_col, merr_col)
            mask = np.logical_and(phot_statistics[:,0] > 0.0, phot_statistics[:,1] > 0.0)
            for star in range(0,len(phot_statistics),1):
                print('Stats: ',phot_statistics[star,0],phot_statistics[star,1])

            fig,ax = plt.subplots(1,1,figsize=(10,10))
            plt.rcParams.update({'font.size': 18})

            mask = np.logical_and(phot_statistics[:,0] > 0.0, phot_statistics[:,1] > 0.0)
            ax.plot(phot_statistics[mask,0], phot_statistics[mask,1], 'k.',
                    marker=".", markersize=1.0, alpha=1.0, label='Weighted RMS')
            print('MASK: ',mask,len(mask),filter, quad, iter)
            mask = np.logical_and(mag_binned > 0.0, rms_binned > 0.0)
            ax.plot(mag_binned[mask], rms_binned[mask], 'rx', label='Binned data')

            rms_binned_min = rms_binned[mask] - rms_std_binned[mask]
            rms_binned_max = rms_binned[mask] + rms_std_binned[mask]
            ax.fill_between(mag_binned[mask], rms_binned_min, rms_binned_max, alpha=0.2)

            ax.set_yscale('log')
            ax.set_xlabel('Weighted mean mag')
            ax.set_ylabel('RMS [mag]')

            plt.grid()
            l = plt.legend()
            plt.tight_layout()

            [xmin,xmax,ymin,ymax] = plt.axis()
            plt.axis([xmin,xmax,1e-3,5.0])

            plot_file = path.join(params['red_dir'],
                            'rms_primary_ref_'+str(filter)+'_quad'+str(quad)+'_iter'+str(iter)+'.png')
            plt.savefig(plot_file)

            log.info('Output binned RMS plot to '+plot_file)
            plt.close(fig)


def calc_rms(phot_data_filter, mag_col, merr_col):
    """Calculate RMS vs mean_mag for all stars in selection"""

    phot_statistics = np.zeros( (len(phot_data_filter),4) )
    (phot_statistics[:,0], phot_statistics[:,3]) = plot_rms.calc_weighted_mean_2D(phot_data_filter, mag_col, merr_col)
    phot_statistics[:,1] = plot_rms.calc_weighted_rms(phot_data_filter, phot_statistics[:,0], mag_col, merr_col)
    phot_statistics[:,2] = plot_rms.calc_percentile_rms(phot_data_filter, phot_statistics[:,0], mag_col, merr_col)

    return phot_statistics

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
