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
from astropy.table import Table, Column
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

    # Identify the datasets to be used as the primary reference in each
    # filter:
    xmatch.id_primary_datasets_per_filter()
    log.info('Identified datasets to be used as the primary references in each filter: '\
                +repr(xmatch.reference_datasets))

    # Add columns to the dataset Table to hold the photometric calibration
    # parameters
    (ndset, ncol) = xmatch.datasets.shape
    xmatch.datasets.add_column(np.zeros(ndset), name='norm_a0', index=ncol+1)
    xmatch.datasets.add_column(np.zeros(ndset), name='norm_a1', index=ncol+2)
    xmatch.datasets.add_column(np.zeros(ndset), name='norm_covar_0', index=ncol+3)
    xmatch.datasets.add_column(np.zeros(ndset), name='norm_covar_1', index=ncol+4)
    xmatch.datasets.add_column(np.zeros(ndset), name='norm_covar_2', index=ncol+5)
    xmatch.datasets.add_column(np.zeros(ndset), name='norm_covar_3', index=ncol+6)
    log.info('Expanded xmatch.datasets table for normalization parameters')

    # Extract list of filters from xmatch.images['filter'] column
    filter_list = np.unique(xmatch.images['filter'].data)
    log.info('Identified list of filters to process: '+repr(filter_list))

    # Read in data from all quadrants into a single array; the photometric
    # calibration should not differ between quadrants.
    log.info('Loading the timeseries photometry from all four quadrants')
    phot_data = hd5_utils.load_four_quadrant_photometry(params['red_dir'],
                                                        params['field_name'])
    log.info('-> Completed photometry load')

    # Identify constant stars in the dataset
    constant_stars = find_constant_stars(xmatch, phot_data)
    log.info('Identified '+str(len(constant_stars))
            +' low-scatter stars to use for the normalization')

    # Normalize the photometry of each dataset to that of the reference
    # image in the primary reference dataset in that filter
    for filter in filter_list:

        # Plot an RMS diagram of the lightcurves for all stars in this filter,
        # prior to normalization, for comparison
        image_index = np.where(xmatch.images['filter'] == filter)[0]
        phot_data_filter = phot_data[:,image_index,:]
        (mag_col, mag_err_col) = plot_rms.get_photometry_columns('corrected')
        plot_multisite_rms(params, phot_data_filter, mag_col, merr_col,
                            'rms_prenorm_'+str(filter)+'.png')

        # Extract the reference image photometry for the primary-ref dataset
        # for this filter
        ref_datacode = xmatch.reference_datasets[filter]
        ref_phot = np.zeros((len(xmatch.stars),2))
        ref_phot[:,0] = xmatch.stars['cal_'+filter+'_mag_'+ref_datacode]
        ref_phot[:,1] = xmatch.stars['cal_'+filter+'_magerr_'+ref_datacode]

        # Extract the lightcurves for all other datasets in this filter in turn
        dataset_index = np.where(self.datasets['filter'] == filter)[0]

        for idset in dataset_index:
            dset_datacode = self.datasets['dataset_code'][idset]
            image_index = np.where(xmatch.images['dataset_code'] == dset_datacode)[0]
            dset_phot[:,0] = xmatch.stars['cal_'+filter+'_mag_'+dset_datacode]
            dset_phot[:,1] = xmatch.stars['cal_'+filter+'_magerr_'+dset_datacode]

            # Calculate their weighted offset relative to the primary-ref
            # dataset for the filter
            (fit, covar_fit) = calc_phot_normalization(ref_phot, dset_phot, constant_stars,
                                                diagnostics=True, ref=ref_datacode,
                                                dset=dset_datacode, f=filter)

            # Store the fit results for this dataset
            xmatch = store_dataset_phot_normalization(idset, xmatch, fit, covar_fit)

            # Apply the normalization calibration to the dataset's reference
            # image photometry, and store the results in the xmatch.stars table
            cal_phot = apply_phot_normalization_single_frame(fit, covar_fit, dset_phot,
                                                                0, 1, log)
            xmatch.stars['norm_'+filter+'_mag_'+dset_datacode] = cal_phot[:,0]
            xmatch.stars['norm_'+filter+'_magerr_'+dset_datacode] = cal_phot[:,1]

            # Apply the photometry calibration to the timeseries data
            # for this dataset
            (mag_col, mag_err_col) = plot_rms.get_photometry_columns('corrected')
            (norm_mag_col, norm_mag_err_col) = plot_rms.get_photometry_columns('normalized')
            phot_data = normalize_timeseries_photometry(phot_data, image_index,
                                                        fit, covar_fit,
                                                        mag_col, mag_err_col,
                                                        norm_mag_col, norm_mag_err_col,
                                                        log)

            # Plot a second RMS diagram of the lightcurves for all stars in this
            # filter, post normalization, for comparison
            image_index = np.where(xmatch.images['filter'] == filter)[0]
            phot_data_filter = phot_data[:,image_index,:]
            (mag_col, mag_err_col) = plot_rms.get_photometry_columns('normalized')
            plot_multisite_rms(params, phot_data_filter, mag_col, merr_col,
                                'rms_postnorm_'+str(filter)+'.png')

    # Output updated crossmatch table and photometry HDF5 files
    xmatch.save(params['crossmatch_file'])
    field_photometry.output_field_photometry(params, xmatch, phot_data, log)

    logs.close_log(log)

    status = 'OK'
    report = 'Completed successfully'
    return status, report

def plot_multisite_rms(params, phot_data, mag_col, merr_col, plot_filename):
    """Function to plot an RMS diagram, with lightcurves combining data from
    multiple sites.  The function expects to receive a pre-filtered array
    of photometry data, for example including lightcurves for all stars
    in a single filter"""

    # Plot a multi-site initial RMS diagram for reference
    phot_statistics = np.zeros( (len(phot_data),4) )
    (wmean,werror) = plot_rms.calc_weighted_mean_2D(phot_data, mag_col, merr_col)

    tmp = plot_rms.calc_weighted_rms(phot_data_filter, phot_statistics[:,0], mag_col, merr_col)

    plot_rms.plot_rms(phot_statistics, params, log)
    rename(path.join(params['red_dir'],'rms.png'),
           path.join(params['red_dir'], plot_filename))


def find_constant_stars(xmatch, phot_data):
    """Identify constant stars from the timeseries photometry of the primary
    reference dataset, by searching for stars in the brightest quartile
    with low RMS scatter."""

    # Identify the overall primary reference dataset:
    ref_dset_idx = np.where(xmatch.datasets['primary_ref'] == 1)[0]
    ref_datacode = xmatch.datasets['dataset_code'][ref_dset_idx]

    # Fetch the indices of the images from this dataset in the photometry table
    image_index = np.where(xmatch.images['dataset_code'] == ref_datacode)[0]

    # Extract the timeseries photometry for this dataset:
    (mag_col, merr_col) = plot_rms.get_photometry_columns('corrected')
    ref_phot = np.zeros((phot_data.shape[0],len(image_index),2))
    ref_phot[:,:,0] = phot_data[:,image_index,mag_col]
    ref_phot[:,:,1] = phot_data[:,image_index,merr_col]

    # Evaluate the photometric scatter of all stars, and select those with
    # the lowest scatter for the brightest quartile of stars.
    (mean_mag, mean_magerr) = calc_weighted_mean(ref_phot)
    rms = calc_weighted_rms(ref_phot, mean_mag)

    # Function identifies constant stars as those with an RMS in the lowest
    # 1 - 25% of the set.  This excludes both stars with high scatter and those
    # with artificially low scatter due to having few measurements.
    rms_range = rms.max() - rms.min()
    min_cut = rms.min()
    max_cut = rms.min() + (rms_range)*0.25

    constant_stars = np.where((rms >= min_cut) & (rms <= max_cut))[0]

    return constant_stars

def store_dataset_phot_normalization(idset, xmatch, fit, covar_fit):
    """Function to store the coefficients and covarience of the photometric
    normalization for this dataset in the datasets table"""

    xmatch.datasets['norm_a0'][idset] = fit[0]
    xmatch.datasets['norm_a1'][idset] = fit[1]
    xmatch.datasets['norm_covar_0'][idset] = fit[0]
    xmatch.datasets['norm_covar_1'][idset] = fit[1]
    xmatch.datasets['norm_covar_2'][idset] = fit[2]
    xmatch.datasets['norm_covar_3'][idset] = fit[3]

    return xmatch

def calc_phot_normalization(ref_phot, dset_phot, constant_stars,
                            diagnostics=False, ref=None, dset=None, f=None):
    """Function to calculate the magnitude offset between the magnitudes of
    constant stars in the given dataset relative to the primary reference.
    """
    fit = np.array([1,0])
    covar_fit = np.zeros((3,3))
    (fit,covar_fit) = calibrate_photometry.calc_transform(fit,
                                                    ref_phot[constant_stars,0],
                                                    dset_phot[constant_stars,0])

    if diagnostics:
        fig = plt.figure(1)
        plt.errorbar(ref_phot[constant_stars,0],
                     dset_phot[constant_stars,0],
                     xerr=ref_phot[constant_stars,1],
                     yerr=dset_phot[constant_stars,1],
                     color='m', fmt='none')
        xplot = np.linspace(ref_phot[constant_stars,0].min(),
                            ref_phot[constant_stars,0].max(), 50)
        yplot = phot_func(fit,xplot)
        plt.plot(xplot, yplot,'k-')
        plt.xlabel('Primary ref magnitudes')
        plt.ylabel('Dataset magnitudes')
        plt.title('Normalization of '+dset_code+' to '+ref_code+' in '+f)
        [xmin,xmax,ymin,ymax] = plt.axis()
        plt.axis([xmax,xmin,ymax,ymin])
        plt.savefig(os.path.join(params['red_dir'],
                    'phot_norm_transform_'+dset_code+'_'+ref_code+'_'+f+'.png'))
        plt.close(1)

    return fit, covar_fit

def apply_phot_normalization_single_frame(fit, covar_fit, frame_phot_data,
                                            mag_col, mag_err_col, log):
    """This function applies the photometric normalization to the photometry
    for a single frame.  While acknowledged to be somewhat slower this ensures
    existing code (calibrate_photometry.calc_calibrated_mags) can be reused. """

    # Extract 2D arrays of the photometry of all stars in all frames,
    # and mask invalid values
    valid = np.logical_and(frame_phot_data[:,mag_col] > 0.0, \
                           frame_phot_data[:,mag_err_col] > 0.0)
    invalid = np.invert(valid)
    nstars = frame_phot_data.shape[0]

    # Build a holding catalog of the photometry in the appropriate format
    star_cat = Table([Column(name='mag', data=frame_phot_data[:,mag_col]),
                      Column(name='mag_err', data=frame_phot_data[:,mag_err_col]),
                      Column(name='cal_ref_mag', data=np.zeros(nstars)),
                      Column(name='cal_ref_mag_err', data=np.zeros(nstars)),
                      Column(name='cal_ref_flux', data=np.zeros(nstars)),
                      Column(name='cal_ref_flux_err', data=np.zeros(nstars))])

    star_cat = calibrate_photometry.calc_calibrated_mags(fit, covar_fit,
                                                         star_cat, log)
    (star_cat['cal_ref_flux'], star_cat['cal_ref_flux_err']) = photometry.convert_mag_to_flux(star_cat['cal_ref_mag'],
                                                                                              star_cat['cal_ref_mag_err'])

    cal_data = np.zeros((frame_phot_data.shape[0],4))
    cal_data[valid,0] = star_cat['cal_ref_mag'][valid]
    cal_data[valid,1] = star_cat['cal_ref_mag_err'][valid]
    cal_data[valid,2] = star_cat['cal_ref_flux'][valid]
    cal_data[valid,3] = star_cat['cal_ref_flux_err'][valid]

    return cal_data

def normalize_timeseries_photometry(phot_data, image_index, fit, covar_fit,
                                    mag_col, mag_err_col,
                                    norm_mag_col, norm_mag_err_col,
                                    log):
    """Function to apply the photometric normalization to the photometry for all
    images from a given dataset.
    phot_data : full array of timeseries data for the quadrant
    image_index : indices of the images of the data from this dataset in the phot_data array
    fit : Two-parameter photometric calibration parameters
    covar_fit : four-parameter covarience of the photometric calibration
    mag_col : column index of the mag photometry values to be calibrated
    mag_err_col : column index of the mag photometric uncertainty values to be calibrated
    norm_mag_col : column index of the mag photometry values to be calibrated
    norm_mag_err_col : column index of the mag photometric uncertainty values to be calibrated
    log : log object for this process
    """

    for i in image_index:
        frame_phot_data = np.zeros((phot_data.shape[0],2))
        frame_phot_data[:,0] = phot_data[:,i,mag_col]
        frame_phot_data[:,1] = phot_data[:,i,mag_err_col]

        cal_phot = apply_phot_normalization_single_frame(fit, covar_fit,
                                                         frame_phot_data,
                                                         0, 1,
                                                         log)

        phot_data[:,i,norm_mag_col] = cal_phot[:,0]
        phot_data[:,i,norm_mag_err_col] = cal_phot[:,1]

    return phot_data

def calc_weighted_mean(data):

    mask = np.invert(np.logical_and(data[:,:,0] > 0.0, data[:,:,1] > 0.0))
    mags = np.ma.array(data[:,:,0], mask=mask)
    errs = np.ma.array(data[:,:,1], mask=mask)

    idx = np.where(mags > 0.0)
    err_squared_inv = 1.0 / (errs*errs)
    wmean =  (mags * err_squared_inv).sum(axis=1) / (err_squared_inv.sum(axis=1))
    werror = np.sqrt( 1.0 / (err_squared_inv.sum(axis=1)) )

    return wmean, werror

def calc_weighted_rms(data, mean_mag):

    mask = np.invert(np.logical_and(data[:,:,0] > 0.0, data[:,:,1] > 0.0))
    mags = np.ma.array(data[:,:,0], mask=mask)
    errs = np.ma.array(data[:,:,1], mask=mask)

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
