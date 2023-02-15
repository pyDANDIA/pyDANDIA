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
    ndset = len(xmatch.datasets)
    ncol = len(xmatch.datasets.colnames)
    if 'norm_a0' not in xmatch.datasets.colnames:
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

    # Read data from quadrant 1
    # Reading in the timeseries data for all four quadrants is at the very
    # edge of the memory limits on the machines available, so it is preferable
    # to calibrate the quadrant's data separately.  However, there are sufficient
    # stars in each quantant to be able to determine the photometric calibration
    # from a single quadrant, and apply it to the rest of the image.
    log.info('Loading the timeseries photometry from quadrant 1')
    file_path = path.join(setup.red_dir, params['field_name']+'_quad1_photometry.hdf5')
    phot_data = hd5_utils.read_phot_from_hd5_file(file_path, return_type='array')
    log.info('-> Completed photometry load')

    # Identify constant stars in the dataset
    constant_stars = find_constant_stars(xmatch, phot_data, log)
    star = 1

    # Normalize the photometry of each dataset to that of the reference
    # image in the primary reference dataset in that filter
    #for filter in filter_list:
    for filter in ['ip']:

        # Plot an RMS diagram of the lightcurves for all stars in this filter,
        # prior to normalization, for comparison
        image_index = np.where(xmatch.images['filter'] == filter)[0]
        phot_data_filter = phot_data[:,image_index,:]
        (mag_col, mag_err_col) = field_photometry.get_field_photometry_columns('corrected')

        plot_multisite_rms(params, phot_data_filter, mag_col, mag_err_col,
                            'rms_prenorm_'+str(filter)+'.png', log)

        # Extract the reference image photometry for the primary-ref dataset
        # for this filter
        ref_datacode = xmatch.reference_datasets[filter]
        sitecode = get_site_code(ref_datacode)
        log.info('Reference dataset in '+filter+' is '+ref_datacode+', sitecode='+sitecode)

        ref_phot = np.zeros((len(xmatch.stars),2))
        ref_phot[:,0] = xmatch.stars['cal_'+filter.replace('p','')+'_mag_'+sitecode]
        ref_phot[:,1] = xmatch.stars['cal_'+filter.replace('p','')+'_magerr_'+sitecode]

        # Extract the lightcurves for all other datasets in this filter in turn
        dataset_index = np.where(xmatch.datasets['dataset_filter'] == filter)[0]

        for idset in dataset_index:
            dset_datacode = xmatch.datasets['dataset_code'][idset]
            dset_sitecode = get_site_code(dset_datacode)

            # If the dataset is the reference dataset, replicate the photometric
            # measurements from the corrected columns to the normalized columns,
            # since no normalization is required - this ensures the full
            # lightcurve can be accessed from the normalization columns.
            if dset_datacode == ref_datacode:
                log.info('Replicating primary reference photometry from dataset '\
                        +dset_datacode+' to the normalized photometry columns')
                image_index = np.where(xmatch.images['dataset_code'] == dset_datacode)[0]
                (mag_col, mag_err_col) = field_photometry.get_field_photometry_columns('corrected')
                (norm_mag_col, norm_mag_err_col) = field_photometry.get_field_photometry_columns('normalized')
                for i in image_index:
                    phot_data[:,i,norm_mag_col] = phot_data[:,i,mag_col]
                    phot_data[:,i,norm_mag_err_col] = phot_data[:,i,mag_err_col]

            # Normalize any dataset that isn't the same as the reference dataset
            else:
                log.info('Normalizing dataset '+dset_datacode+', sitecode='+dset_sitecode)
                image_index = np.where(xmatch.images['dataset_code'] == dset_datacode)[0]
                dset_phot = np.zeros((len(xmatch.stars),2))
                dset_phot[:,0] = xmatch.stars['cal_'+filter.replace('p','')+'_mag_'+dset_sitecode]
                dset_phot[:,1] = xmatch.stars['cal_'+filter.replace('p','')+'_magerr_'+dset_sitecode]

                # Calculate their weighted offset relative to the primary-ref
                # dataset for the filter
                (fit, covar_fit) = calc_phot_normalization(ref_phot, dset_phot,
                                                            constant_stars, log,
                                                    diagnostics=True, ref=sitecode,
                                                    dset=dset_sitecode, f=filter)

                # Store the fit results for this dataset
                xmatch = store_dataset_phot_normalization(idset, xmatch, fit, covar_fit, log)

                # Apply the normalization calibration to the dataset's reference
                # image photometry, and store the results in the xmatch.stars table
                log.info('Applying normalization to the datasets reference image photometry')
                cal_phot = apply_phot_normalization_single_frame(fit, covar_fit, dset_phot,
                                                                    0, 1, log,
                                                            diagnostics=True, ref=sitecode,
                                                            dset=dset_sitecode, f=filter)
                xmatch.stars['norm_'+filter.replace('p','')+'_mag_'+dset_sitecode] = cal_phot[:,0]
                xmatch.stars['norm_'+filter.replace('p','')+'_magerr_'+dset_sitecode] = cal_phot[:,1]

                # Apply the photometry calibration to the timeseries data
                # for this dataset
                (mag_col, mag_err_col) = field_photometry.get_field_photometry_columns('corrected')
                (norm_mag_col, norm_mag_err_col) = field_photometry.get_field_photometry_columns('normalized')
                phot_data = normalize_timeseries_photometry(phot_data, image_index,
                                                            fit, covar_fit,
                                                            mag_col, mag_err_col,
                                                            norm_mag_col, norm_mag_err_col,
                                                            log)

        # Plot a second RMS diagram of the lightcurves for all stars in this
        # filter, post normalization, for comparison
        image_index = np.where(xmatch.images['filter'] == filter)[0]
        phot_data_filter = phot_data[:,image_index,:]
        (mag_col, mag_err_col) = field_photometry.get_field_photometry_columns('normalized')
        plot_multisite_rms(params, phot_data_filter, mag_col, mag_err_col,
                            'rms_postnorm_'+str(filter)+'.png', log)


        fig = plt.figure(3,(10,10))
        (norm_mag_col, norm_mag_err_col) = field_photometry.get_field_photometry_columns('normalized')
        idx = np.where(phot_data[star,:,norm_mag_col] > 0.0)[0]
        plt.errorbar(phot_data[star,idx,0], phot_data[star,idx,norm_mag_col],
                     yerr=phot_data[star,idx,norm_mag_err_col], fmt='none', color='k')
        (xmin,xmax,ymin,ymax) = plt.axis()
        ymin = max(ymin,14.0)
        ymax = min(ymax,22.0)
        plt.axis([xmin,xmax,ymax,ymin])
        plt.xlabel('HJD')
        plt.ylabel('Mag')
        plt.savefig('Star_'+str(star)+'_lc_norm.png')
        plt.close(3)

    # Output updated crossmatch table
    xmatch.save(params['crossmatch_file'])

    # Output the photometry for quadrant 1:
    output_quadrant_photometry(params, setup, 1, phot_data, log)

    logs.close_log(log)

    status = 'OK'
    report = 'Completed successfully'
    return status, report

def output_quadrant_photometry(params, setup, qid, photometry, log):

    filename = params['field_name']+'_quad'+str(qid)+'_photometry.hdf5'
    log.info('Outputting photometry for quadrant '+str(qid)
                +', array shape: '+repr(photometry.shape)
                +' to '+filename)

    hd5_utils.write_phot_hd5(setup, photometry, log=log,
                                filename=filename)

def get_site_code(datacode):
    entries = datacode.split('_')[1].split('-')
    sitecode = entries[0]+'_'+entries[1]
    return sitecode

def plot_multisite_rms(params, phot_data, mag_col, merr_col, plot_filename, log):
    """Function to plot an RMS diagram, with lightcurves combining data from
    multiple sites.  The function expects to receive a pre-filtered array
    of photometry data, for example including lightcurves for all stars
    in a single filter"""

    # Plot a multi-site initial RMS diagram for reference
    phot_statistics = np.zeros( (len(phot_data),4) )
    (phot_statistics[:,0],werror) = plot_rms.calc_weighted_mean_2D(phot_data, mag_col, merr_col)
    phot_statistics[:,1] = plot_rms.calc_weighted_rms(phot_data, phot_statistics[:,0], mag_col, merr_col)

    text_filename = plot_filename.replace('.png', '.txt')
    f = open(path.join(params['red_dir'],text_filename),'w')
    f.write('# Star_index  wMean_mag   RMS\n')
    for j in range(0,len(phot_statistics),1):
        f.write(str(j)+' '+str(phot_statistics[j,0])+' '+str(phot_statistics[j,1])+'\n')
    f.close()

    fig = plt.figure(3,(10,10))
    plt.rcParams.update({'font.size': 18})
    mask = np.logical_and(phot_statistics[:,0] > 0.0, phot_statistics[:,1] > 0.0)
    plt.plot(phot_statistics[mask,0], phot_statistics[mask,1], 'k.',
            markersize=0.5, label='Weighted RMS')
    plt.yscale('log')
    plt.xlabel('Weighted mean mag')
    plt.ylabel('RMS [mag]')
    plt.grid()
    l = plt.legend()
    plt.tight_layout()
    [xmin,xmax,ymin,ymax] = plt.axis()
    plt.axis([xmin,xmax,1e-3,5.0])
    plot_file = path.join(params['red_dir'],plot_filename)
    plt.savefig(plot_file)
    log.info('Output RMS plot to '+plot_filename)
    plt.close(3)

def find_constant_stars(xmatch, phot_data, log):
    """Identify constant stars from the timeseries photometry of the primary
    reference dataset, by searching for stars in the brightest quartile
    with low RMS scatter."""

    # Identify the overall primary reference dataset:
    ref_dset_idx = np.where(xmatch.datasets['primary_ref'] == 1)[0]
    ref_datacode = xmatch.datasets['dataset_code'][ref_dset_idx]

    # Fetch the indices of the images from this dataset in the photometry table
    image_index = np.where(xmatch.images['dataset_code'] == ref_datacode)[0]

    # Extract the timeseries photometry for this dataset:
    (mag_col, merr_col) = field_photometry.get_field_photometry_columns('corrected')
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

    log.info('Identified '+str(len(constant_stars))
            +' stars with RMS between '+str(round(min_cut,3))+' and '+str(round(max_cut,3))
            +'mag to use for the normalization')

    return constant_stars

def store_dataset_phot_normalization(idset, xmatch, fit, covar_fit, log):
    """Function to store the coefficients and covarience of the photometric
    normalization for this dataset in the datasets table"""

    xmatch.datasets['norm_a0'][idset] = fit[0]
    xmatch.datasets['norm_a1'][idset] = fit[1]
    xmatch.datasets['norm_covar_0'][idset] = covar_fit[0,0]
    xmatch.datasets['norm_covar_1'][idset] = covar_fit[0,1]
    xmatch.datasets['norm_covar_2'][idset] = covar_fit[1,0]
    xmatch.datasets['norm_covar_3'][idset] = covar_fit[1,1]

    log.info('Stored the normalization parameters in the crossmatch.datasets table:')
    log.info(repr(xmatch.datasets))

    return xmatch

def fetch_dataset_phot_normalization(idset, xmatch, log):

    fit = np.array([xmatch.datasets['norm_a0'][idset],
                    xmatch.datasets['norm_a1'][idset]])
    covar_fit = np.array([ [xmatch.datasets['norm_covar_0'][idset],
                            xmatch.datasets['norm_covar_1'][idset]],
                           [xmatch.datasets['norm_covar_2'][idset],
                            xmatch.datasets['norm_covar_3'][idset]] ])
    log.info('Loaded the normalization parameters from the crossmatch.datasets table:')
    log.info('Fit: '+repr(fit))
    log.info('Co-varience: '+repr(covar_fit))

    return fit, covar_fit

def calc_phot_normalization(ref_phot, dset_phot, constant_stars, log,
                            diagnostics=False, ref=None, dset=None, f=None):
    """Function to calculate the magnitude offset between the magnitudes of
    constant stars in the given dataset relative to the primary reference.
    """
    fit = np.array([1,0])
    covar_fit = np.zeros((3,3))

    # Select from the list of constant stars in the primary reference in i-band,
    # those which have valid photometric measurements in the current dataset
    # and filter
    # Also require that the difference between the dataset and primary reference
    # magnitude does not exceed a given threshold.  This is to stop a large
    # number of very faint stellar detections being overweighted in the fit,
    # compared with the smaller number of better-measured brighter stars.
    delta_mag = abs(ref_phot[constant_stars,0] - dset_phot[constant_stars,0])
    valid = ((ref_phot[constant_stars,0] > 0.0)
                & (ref_phot[constant_stars,1] <= 0.05)
                & (dset_phot[constant_stars,0] > 0.0)
                & (dset_phot[constant_stars,1] <= 0.05)
                & (delta_mag <= 0.5))

    (fit,covar_fit) = calibrate_photometry.calc_transform(fit,
                                                    dset_phot[constant_stars[valid],0],
                                                    ref_phot[constant_stars[valid],0])
    log.info('Normalization calibration fit parameters: '+repr(fit))
    log.info('Covarience: '+repr(covar_fit))

    if diagnostics:
        fig = plt.figure(1,(10,10))
        plt.errorbar(dset_phot[constant_stars,0],
                     ref_phot[constant_stars,0],
                     xerr=ref_phot[constant_stars,1],
                     yerr=dset_phot[constant_stars,1],
                     color='k', fmt='none', label='Constant stars')
        plt.errorbar(dset_phot[constant_stars[valid],0],
                      ref_phot[constant_stars[valid],0],
                      xerr=ref_phot[constant_stars[valid],1],
                      yerr=dset_phot[constant_stars[valid],1],
                      color='m', fmt='none', label='Valid calibrators')

        xrange = set_plot_range(dset_phot[constant_stars,0])
        yrange = set_plot_range(ref_phot[constant_stars,0])
        xplot = np.linspace(xrange[0], xrange[1], 50)
        yplot = calibrate_photometry.phot_func(fit,xplot)
        plt.plot(xplot, yplot,'k-')
        plt.xlabel('Dataset magnitudes')
        plt.ylabel('Primary ref magnitudes')
        plt.legend()
        plt.grid()
        plt.title('Normalization of '+dset+' to '+ref+' in '+f)
        [xmin,xmax,ymin,ymax] = plt.axis()
        plt.axis([xrange[0],xrange[1],yrange[0],yrange[1]])
        plt.savefig(path.join(params['red_dir'],
                    'phot_norm_transform_'+dset+'_'+ref+'_'+f+'.png'))
        plt.close(1)

        f = open(path.join(params['red_dir'],
                    'phot_norm_transform_'+dset+'_'+ref+'_'+f+'.dat'),'w')
        f.write('# Star_index  '+ref+'_mag '+ref+'_mag_err '+dset+'_mag '+dset+'_mag_err\n')
        for j in constant_stars[valid]:
            f.write(str(j)+' '+str(ref_phot[j,0])+' '+str(ref_phot[j,1])+' '\
                    +str(dset_phot[j,0])+' '+str(dset_phot[j,1])+'\n')
        f.close()

    return fit, covar_fit

def set_plot_range(phot_array):
    idx = np.where(phot_array > 0.0)[0]
    return [phot_array[idx].min(), phot_array[idx].max()]

def apply_phot_normalization_single_frame(fit, covar_fit, frame_phot_data,
                                            mag_col, mag_err_col, log,
                                            diagnostics=False,
                                            ref=None, dset=None, f=None):
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

    if diagnostics:
        file_path = open(path.join(params['red_dir'],
                    'norm_phot_'+dset+'_'+ref+'_'+f+'.txt'),'w')
        file_path.write('# Star_id  cal_mag cal_merr  norm_mag  norm_merr\n')
        for j in valid:
            file_path.write(str(j)+' '+str(frame_phot_data[valid,mag_col])+' '\
                        +str(frame_phot_data[valid,mag_err_col])+' '\
                        +str(cal_data[valid,0])+' '+str(cal_data[valid,1])+'\n')
        file_path.close()

        fig = plt.figure(2,(10,10))
        plt.plot(frame_phot_data[valid,mag_col], frame_phot_data[valid,mag_err_col],
                'g.', label='Pre-calibration')
        plt.plot(cal_data[valid,0], cal_data[valid,1], 'm.', label='Normalized')
        plt.xlabel('Magnitude')
        plt.ylabel('Magnitude error')
        plt.yscale('log')
        plt.grid()
        (xmin,xmax,ymin,ymax) = plt.axis()
        xmin = max(14.0, xmin)
        xmax = min(22.0, xmax)
        plt.axis([xmin,xmax,ymin,ymax])
        plt.legend()
        plt.savefig(path.join(params['red_dir'],
                    'norm_phot_'+dset+'_'+ref+'_'+f+'.png'))
        plt.close(2)

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

    log.info('Normalizing timeseries photometry for '+str(len(image_index))+' images')

    for i in image_index:
        frame_phot_data = np.zeros((phot_data.shape[0],2))
        frame_phot_data[:,0] = phot_data[:,i,mag_col]
        frame_phot_data[:,1] = phot_data[:,i,mag_err_col]
        if i%10==0: log.info(' -> Image '+str(i))

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
