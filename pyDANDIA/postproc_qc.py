from os import path, rename
from sys import argv
import numpy as np
from pyDANDIA import crossmatch
from pyDANDIA import hd5_utils
from pyDANDIA import logs
from pyDANDIA import metadata
from pyDANDIA import plot_rms
from pyDANDIA import pipeline_setup
from pyDANDIA import config_utils
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from astropy import visualization
from astropy.io import fits
import glob

def run_postproc(setup, **params):
    """Driver function for post-processing:
    Assessment of photometric residuals and uncertainties
    """

    log = logs.start_stage_log( setup.red_dir, 'postproc_phot' )
    (setup, params) = sanity_check(setup, params,log)

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')
    phot_file = path.join(setup.red_dir,'photometry.hdf5')
    photometry = hd5_utils.read_phot_from_hd5_file(phot_file, return_type='array')
    log.info('Loaded dataset photometry and metadata')

    # Grow photometry array to allow additional columns for corrected mags
    photometry = grow_photometry_array(photometry,log)
    photometry = mask_photometry_array(photometry, 1, log)

    # Calculate mean_mag, RMS for all stars
    phot_stats = plot_rms.calc_mean_rms_mag(photometry,log,'calibrated')
    plot_rms.plot_rms(phot_stats, params, log,
                    plot_file=path.join(setup.red_dir,'init_rms_mag.png'))
    plot_rms.output_phot_statistics(phot_stats,
                                    path.join(setup.red_dir,'init_rms_mag.txt'),
                                    log)

    # Calculate photometric residuals
    phot_residuals = calc_phot_residuals(photometry, phot_stats, log, 'calibrated')
    plot_phot_residuals(params, reduction_metadata, phot_residuals, log)

    # Calculate mean residual per image
    image_residuals = calc_image_residuals(reduction_metadata, photometry, phot_residuals,log)
    plot_image_residuals(params, image_residuals, log)

    # Mask photometry from images with excessive average photometric residuals
    photometry = mask_phot_from_bad_images(params, photometry, image_residuals, 2, log)

    # Mask photometry from datapoints which fail the ps/exptime criterion
    photometry = mask_phot_with_bad_psexpt(params, reduction_metadata, photometry, 4, log)

    # Mask photometry from images with unreliable resampling coefficients:
    photometry = mask_phot_from_bad_warp_matrix(params, setup, photometry, 8, log)

    # Mask photometry from images with poor quality difference images
    photometry = mask_phot_from_bad_diff_images(params,setup,photometry,16,log)

    # Mirror calibrated photometry to corrected columns ready for processing:
    photometry = mirror_mag_columns(photometry, 'calibrated', log)

    # Compensate for mean residual per image and output to photometry array
    #photometry = apply_image_mag_correction(params, image_residuals, photometry, log, 'calibrated')

    # Re-calculate mean_mag, RMS for all stars, using corrected magnitudes
    phot_stats = plot_rms.calc_mean_rms_mag(photometry,log,'corrected')
    phot_stats = mask_photometry_stats(phot_stats, log)
    plot_rms.plot_rms(phot_stats, params, log,
                    plot_file=path.join(setup.red_dir,'postproc_rms_mag.png'))
    plot_rms.output_phot_statistics(phot_stats,
                path.join(setup.red_dir,'postproc_rms_mag.txt'),
                log)

    # Re-calculate photometric residuals
    phot_residuals = calc_phot_residuals(photometry, phot_stats, log, 'corrected')
    image_residuals = calc_image_residuals(reduction_metadata, photometry, phot_residuals,log)

    # Calculate photometric scatter per image
    image_rms = calc_image_rms(phot_residuals, log)

    # Factor photometric scatter into photometric residuals
    photometry = apply_image_merr_correction(photometry, image_residuals, log, 'corrected')
    phot_stats = plot_rms.calc_mean_rms_mag(photometry,log,'corrected')

    # Set quality control flags to indicate suspect data points in photometry array
    # Replaced with a bitmask
    #photometry = set_image_photometry_qc_flags(photometry, log)

    # Set quality control flags to indicate datapoints with large uncertainties:
    # Replaced with bitmask, may review
    #photometry = set_star_photometry_qc_flags(photometry, phot_stats, log)

    # Ouput updated photometry
    output_revised_photometry(setup, photometry, log)

    test_plot_lcs(setup, photometry, log)

    log.info('Post-processing: complete')

    logs.close_log(log)

    status = 'OK'
    report = 'Completed successfully'
    return status, report

def get_args():

    params = {}
    if len(argv) == 1:
        params['red_dir'] = input('Please enter the path to the datasets reduction directory: ')
    else:
        params['red_dir'] = argv[1]

    params['log_dir'] = params['red_dir']
    setup = pipeline_setup.pipeline_setup(params)

    config_file = path.join(setup.pipeline_config_dir, 'postproc_config.json')

    config = config_utils.build_config_from_json(config_file)

    for key, value in config.items():
        if key == 'diagnostic_plots':
            if 'true' in str(value).lower():
                params[key] = True
            else:
                params[key] = False
        else:
            params[key] = value

    return setup, params

def sanity_check(setup, params, log):

    for key in ['red_dir', 'log_dir']:
        if key not in params.keys():
            params[key] = getattr(setup,key)

    if 'diagnostic_plots' not in params.keys():
        params['diagnostic_plots'] = False

    log.info('Configuration parameters:')
    for key, value in params.items():
        log.info(key+': '+str(value))

    return setup, params

def grow_photometry_array(photometry,log):

    (mag_col, merr_col) = plot_rms.get_photometry_columns('calibrated')

    if photometry.shape[2] < 26:
        new_photometry = np.zeros((photometry.shape[0], photometry.shape[1], photometry.shape[2]+3))
        new_photometry[:,:,0:photometry.shape[2]] = photometry
        log.info('Added three columns to the photometry array')

        return new_photometry
    else:
        log.info('Photometry array already has all 26 columns, zeroing columns 23,24,25')
        photometry[:,:,23].fill(0.0)
        photometry[:,:,24].fill(0.0)
        photometry[:,:,25].fill(0.0)
        return photometry

def mask_photometry_array(photometry, error_code, log):

    (mag_col, merr_col) = plot_rms.get_photometry_columns('calibrated')

    mask = np.invert(photometry[:,:,mag_col] > 0.0)

    expand_mask = np.empty((mask.shape[0], mask.shape[1], photometry.shape[2]))
    for col in range(0,expand_mask.shape[2],1):
        expand_mask[:,:,col] = mask

    photometry[mask,25] = error_code

    photometry = np.ma.masked_array(photometry, mask=expand_mask)
    log.info('Masked invalid measurements in photometry array')

    return photometry

def mask_photometry_stats(phot_stats, log):

    mask = np.invert(phot_stats[:,0] > 0.0)

    expand_mask = np.empty((mask.shape[0],phot_stats.shape[1]))
    for col in range(0,phot_stats.shape[1],1):
        expand_mask[:,col] = mask

    phot_stats = np.ma.masked_array(phot_stats, mask=expand_mask)
    log.info('Masked invalid data in photometric statistics for each star')

    return phot_stats

def calc_phot_residuals(photometry, phot_stats, log, phot_columns):

    (mag_col, merr_col) = plot_rms.get_photometry_columns(phot_columns)

    # To subtract a 1D vector from a 2D array, we need to add an axis to ensure
    # the correct Python array handling
    mean_mag = phot_stats[:,0, np.newaxis]
    mean_merr = phot_stats[:,3, np.newaxis]

    mask = np.ma.getmask(photometry)
    phot_residuals = np.zeros((photometry.shape[0],photometry.shape[1],2))
    phot_residuals = np.ma.masked_array(phot_residuals, mask=mask[:,:,mag_col:merr_col+1])

    phot_residuals[:,:,0] = photometry[:,:,mag_col] - mean_mag
    phot_residuals[:,:,1] = np.sqrt( photometry[:,:,merr_col]*photometry[:,:,merr_col] + \
                                    mean_merr*mean_merr )

    log.info('Calculated photometric residuals')

    return phot_residuals

def plot_phot_residuals(params, reduction_metadata, phot_residuals, log):

    if params['diagnostic_plots']:
        for i in range(0,phot_residuals.shape[1],1):
            image_name = reduction_metadata.headers_summary[1]['IMAGES'][i]

            stamp_residuals = []
            for stamp in reduction_metadata.stamps[1]:
                xdx = np.logical_and(reduction_metadata.star_catalog[1]['x'] >= float(stamp['X_MIN']),
                                     reduction_metadata.star_catalog[1]['x'] <= float(stamp['X_MAX']))
                ydx = np.logical_and(reduction_metadata.star_catalog[1]['y'] >= float(stamp['Y_MIN']),
                                  reduction_metadata.star_catalog[1]['y'] <= float(stamp['Y_MAX']))
                jdx = np.logical_and(xdx, ydx)

                err_squared_inv = 1.0 / (phot_residuals[jdx,i,1]*phot_residuals[jdx,i,1])
                stamp_residuals.append( (phot_residuals[jdx,i,0] * err_squared_inv).sum(axis=0) / (err_squared_inv.sum(axis=0)) )

            stamp_residuals = np.array(stamp_residuals)
            stamp_residuals = stamp_residuals.reshape(4,4)

            fig = plt.figure(1,(10,10))
            img = plt.imshow(stamp_residuals)
            plt.colorbar(img)
            plt.savefig(path.join(params['red_dir'],'stamp_residuals_image_'+image_name.replace('.fits','')+'.png'))
            plt.close(1)

            jdx = np.where(phot_residuals[:,i,0] > -9999.0)
            #(hist, bins) = np.histogram(phot_residuals[jdx,i,0])
            bins = np.arange(-0.2,0.2,0.01)
            bin_idx = np.digitize(phot_residuals[jdx,i,0], bins)
            bin_idx -= 1

            fig = plt.figure(2,(10,10))
            plt.scatter(reduction_metadata.star_catalog[1]['x'][jdx], reduction_metadata.star_catalog[1]['y'][jdx],
                        c=bins[bin_idx][0,:], cmap='viridis')
            cbar = plt.colorbar(ticks=bins)
            #cbar.ax.set_xticklabels(['Low', 'Medium', 'High'])  # horizontal colorbar
            plt.xlabel('X [pix]')
            plt.xlabel('Y [pix]')
            plt.savefig(path.join(params['red_dir'],'residuals_image_'+image_name.replace('.fits','')+'.png'))
            plt.close(2)

        log.info('Plotted weighted mean of photometric residuals per stamp for all images')
    else:
        log.info('Photometric residual diagnostic plots switched off')

def calc_image_residuals(reduction_metadata, photometry, phot_residuals, log):

    image_residuals = np.zeros((phot_residuals.shape[1],3))

    err_squared_inv = 1.0 / (phot_residuals[:,:,1]*phot_residuals[:,:,1])
    image_residuals[:,0] =  (phot_residuals[:,:,0] * err_squared_inv).sum(axis=0) / (err_squared_inv.sum(axis=0))
    #image_residuals[:,1] = 1.0 / (err_squared_inv.sum(axis=0))
    dmags = phot_residuals[:,:,0] - image_residuals[:,1]
    image_residuals[:,1] =  np.sqrt( (dmags**2 * err_squared_inv).sum(axis=0) / (err_squared_inv.sum(axis=0)) )
    image_residuals[:,2] = photometry[:,:,9].mean(axis=0)
    log.info('Calculated weighted mean photometric residual per image')

    mask = np.ma.getmask(phot_residuals)
    nvalid = np.sum(np.invert(mask[:,:,0]), axis=0)
    mask = (nvalid == 0)
    extend_mask = np.empty(image_residuals.shape, dtype=bool)
    for col in range(0,3,1):
        extend_mask[:,col] = mask
    image_residuals = np.ma.masked_array(image_residuals, mask=extend_mask)
    log.info('Masked image entries where no stars had valid photometric residuals')

    log.info('Image weighted mean photometric residual and RMS:')
    for i, image_name in enumerate(reduction_metadata.headers_summary[1]['IMAGES']):
        log.info(str(i)+' '+image_name+' '+str(image_residuals[i,0])+' '+\
                str(image_residuals[i,1])+' '+str(image_residuals[i,2]))

    return image_residuals

def plot_image_residuals(params, image_residuals, log):

    fig = plt.figure(1,(10,10))
    plt.errorbar(image_residuals[:,2], image_residuals[:,0],
                yerr=image_residuals[:,1],
                color='m', fmt='none')
    plt.plot(image_residuals[:,2],image_residuals[:,0],'m.')
    plt.xlabel('Image index')
    plt.ylabel('Weighted mean photometric residual [mag]')
    plt.savefig(path.join(params['red_dir'],'image_phot_residuals.png'))
    plt.close(1)

    fig = plt.figure(2,(10,10))
    xdata = np.arange(0,len(image_residuals),1)
    plt.errorbar(xdata, image_residuals[:,0],
                yerr=image_residuals[:,1],
                color='m', fmt='none')
    plt.plot(xdata,image_residuals[:,0],'m.')
    (xmin,xmax,ymin1,ymax1) = plt.axis()
    plt.axis([xmin,xmax,-0.1,0.1])
    plt.xlabel('Image index')
    plt.ylabel('Weighted mean photometric residual [mag]')
    plt.savefig(path.join(params['red_dir'],'image_phot_residuals_zoom.png'))
    plt.close(2)

def apply_image_mag_correction(params, image_residuals, photometry, log, phot_columns):

    (mag_col, merr_col) = plot_rms.get_photometry_columns(phot_columns)

    photometry[:,:,photometry.shape[2]-2] = photometry[:,:,mag_col] - image_residuals[:,0]
    photometry[:,:,photometry.shape[2]-1] = np.sqrt( photometry[:,:,merr_col]*photometry[:,:,merr_col] + \
                                                        image_residuals[:,1]*image_residuals[:,1] )

    log.info('Applied magnitude offset to calculate corrected magnitudes')

    return photometry

def test_plot_lcs(setup, photometry, log):

    test_star_idxs = [30001, 78283, 109708, 120495, 166501]

    phot_data = np.ma.getdata(photometry)

    init_lcs = []
    post_lcs = []
    for star in test_star_idxs:
        idx = np.where(phot_data[star,:,9] > 0)[0]
        print(idx, phot_data[star,idx,9])
        import pdb; pdb.set_trace()
        init_lcs.append( phot_data[star,idx,[9,13,14]].T )
        post_lcs.append( phot_data[star,idx,[9,23,24,25]].T )
        #idx = np.where(lc[:,3] == 0)[0]
        #post_lcs.append( lc[idx,:] )

    for j, star in enumerate(test_star_idxs):
        init_lc = init_lcs[j]
        post_lc = post_lcs[j]

        fig = plt.figure(1,(10,10))
        mean_mag = init_lc[:,1].mean()
        ymin2 = mean_mag + 1.0
        ymax2 = mean_mag - 1.0
        with_errors = True
        if with_errors:
            plt.errorbar(init_lc[:,0]-2450000.0,init_lc[:,1],
                    yerr=init_lc[:,2],
                    color='k', fmt='none')
        else:
            plt.plot(init_lc[:,0]-2450000.0,init_lc[:,1],'k.')
        plt.xlabel('HJD-2450000.0')
        plt.ylabel('Mag')
        plt.title('Star '+str(star+1))
        (xmin,xmax,ymin1,ymax1) = plt.axis()
        plt.axis([xmin,xmax,ymin2,ymax2])
        plt.savefig(path.join(setup.red_dir,'test_lightcurve_init_'+str(star+1)+'.png'))
        plt.close(1)

        fig = plt.figure(2,(10,10))
        if with_errors:
            plt.errorbar(post_lc[:,0]-2450000.0,post_lc[:,1],
                    yerr=post_lc[:,2],
                    color='k', fmt='none')
        else:
            plt.plot(post_lc[:,0]-2450000.0,post_lc[:,1],'k.')

        badidx = np.where(post_lc[:,3] != 0)[0]
        print(star+1, badidx)
        if len(badidx) > 0:
            if with_errors:
                plt.errorbar(post_lc[badidx,0]-2450000.0,post_lc[badidx,1],
                        yerr=post_lc[badidx,2],color='m', fmt='none')
            else:
                plt.plot(post_lc[badidx,0]-2450000.0,post_lc[badidx,1],'m.')

        plt.xlabel('HJD-2450000.0')
        plt.ylabel('Mag')
        plt.title('Star '+str(star+1))
        (xmin,xmax,ymin1,ymax1) = plt.axis()
        plt.axis([xmin,xmax,ymin2,ymax2])
        plt.savefig(path.join(setup.red_dir,'test_lightcurve_post_'+str(star+1)+'.png'))
        plt.close(2)

    log.info('Plotted test star lightcurves')

def calc_image_rms(phot_residuals, log):

    err_squared_inv = 1.0 / (phot_residuals[:,:,1]*phot_residuals[:,:,1])
    rms =  np.sqrt( (phot_residuals[:,:,0]**2 * err_squared_inv).sum(axis=0) / (err_squared_inv.sum(axis=0)) )
    error = np.sqrt( 1.0 / (err_squared_inv.sum(axis=0)) )

    log.info('Calculated RMS of photometric residuals per image ')
    log.info('Image   RMS  std.error')
    for i in range(0,len(rms),1):
        log.info(str(i)+' '+str(rms[i])+' '+str(error[i]))
    return error

def apply_image_merr_correction(photometry, image_residuals, log, phot_columns):

    (mag_col, merr_col) = plot_rms.get_photometry_columns(phot_columns)

    photometry[:,:,merr_col] = np.sqrt( photometry[:,:,merr_col]*photometry[:,:,merr_col] + \
                                                        image_residuals[:,0]*image_residuals[:,0] )

    log.info('Factored the image residuals into photometric uncertainties')

    return photometry

def output_revised_photometry(setup, photometry, log):

    # Back up the older photometry file for now
    phot_file_name = path.join(setup.red_dir,'photometry.hdf5')
    bkup_file_name = path.join(setup.red_dir,'photometry_stage6.hdf5')
    if path.isfile(phot_file_name):
        rename(phot_file_name, bkup_file_name)

    # Output file with additional columns:
    hd5_utils.write_phot_hd5(setup,photometry,log=log)

def mask_all_datapoints_by_image_index(photometry, bad_data_index, error_code):

    expand_mask = np.ma.getmask(photometry)
    expand_data = np.ma.getdata(photometry)
    for i in bad_data_index:
        mask = np.empty((photometry.shape[0],photometry.shape[2]), dtype='bool')
        mask.fill(True)
        expand_mask[:,i,:] = mask
        expand_data[:,i,25] += error_code

    photometry = np.ma.masked_array(expand_data, mask=expand_mask)

    return photometry

def mask_phot_from_bad_images(params, photometry, image_residuals, error_code, log):

    idx = np.where(abs(image_residuals[:,0]) > params['residuals_threshold'])[0]

    photometry = mask_all_datapoints_by_image_index(photometry, idx, error_code)

    log.info('Masked photometric data for images with excessive average residuals')

    return photometry

def mask_phot_with_bad_psexpt(params, reduction_metadata, photometry, error_code, log):

    ps_expt = calc_ps_exptime(reduction_metadata, photometry, log)

    idx = np.where(ps_expt < params['psexpt_threshold'])

    mask = np.ma.getmask(photometry)
    data = np.ma.getdata(photometry)
    for col in range(0,mask.shape[2],1):
        mask[idx[0],idx[1],col] = True
    data[idx[0],idx[1],25] += error_code

    photometry = np.ma.masked_array(data, mask=mask)

    log.info('Masked photometric data for datapoints with ps/expt < '+str(params['psexpt_threshold']))

    return photometry

def calc_ps_exptime(reduction_metadata, photometry, log):

    ps_data = photometry[:,:,19]
    mask = np.empty(ps_data.shape)
    mask.fill(False)

    exptimes = reduction_metadata.headers_summary[1]['EXPKEY'].data
    expt_data = np.zeros(ps_data.shape, dtype='float')
    for i in range(0,expt_data.shape[1],1):
        expt_data[:,i].fill(float(exptimes[i]))
    reference_image_name = reduction_metadata.data_architecture[1]['REF_IMAGE'].data[0]
    iref = np.where(reduction_metadata.headers_summary[1]['IMAGES'] == reference_image_name)
    ref_expt = float(exptimes[iref])

    ps_data = np.ma.masked_array(ps_data, mask=mask)
    ps_expt = ps_data*ref_expt/expt_data

    log.info('Calculated the pscale/exptime quality control metric')

    return ps_expt

def load_resampled_data(setup,log):

    frames = []
    coefficients = np.zeros(1)

    resampled_dir = path.join(setup.red_dir, 'resampled')
    if not path.isdir(resampled_dir):
        log.info('Cannot find the directory for resampled data at '+resampled_dir)
        return frames, coefficients

    frame_list = glob.glob(path.join(resampled_dir, '*.fits'))
    if len(frame_list) == 0:
        log.info('No stage 4 output per frame available')
        return frames, coefficients

    coefficients = np.zeros((9,len(frame_list)))
    for i,frame in enumerate(frame_list):
        f = path.join(resampled_dir, frame, 'warp_matrice_stamp_10.npy')
        if not path.isfile(f):
            f = path.join(resampled_dir, frame, 'warp_matrice_stamp_0.npy')

        if path.isfile(f):
            coefficients[:,i] = np.load(f).flatten()

    frames = []
    for f in frame_list:
        frames.append(path.basename(f))

    return frames, coefficients

def mask_phot_from_bad_warp_matrix(params,setup,photometry,error_code,log):

    (frames, coefficients) = load_resampled_data(setup,log)

    idx = np.where(coefficients > params['warp_matrix_threshold'])[0]

    photometry = mask_all_datapoints_by_image_index(photometry, idx, error_code)

    log.info('Masked photometric data for difference images with excessive residuals')

    return photometry

def mask_phot_from_bad_diff_images(params,setup,photometry,error_code,log):

    diff_dir = path.join(setup.red_dir,'diffim')
    diff_images = glob.glob(path.join(diff_dir, '*'))
    diff_images.sort()

    dimage_stats = []
    for dimage_path in diff_images:
        stats = calc_stamp_statistics(params,dimage_path,log)
        dimage_stats.append(stats)
    dimage_stats = np.array(dimage_stats)

    plot_dimage_statistics(params, dimage_stats, diff_images)

    # Use only first dimension of this array, which is images,stamps
    # rather than stars, images
    idx = np.where(dimage_stats[:,:,2] > params['diff_std_threshold'])[0]

    photometry = mask_all_datapoints_by_image_index(photometry, idx, error_code)

    log.info('Masked datapoints from poor quality difference images')

    return photometry

def calc_stamp_statistics(params,dimage_path,log):
    statistics = []

    if params['stamp_number'] == -1:
        stamps = glob.glob(path.join(dimage_path,'diff_stamp_*.fits'))

        for i,stamp in enumerate(stamps):
            image = fits.getdata(stamp)
            statistics.append([i,np.median(image), image.std()])
    else:
        stamp = path.join(dimage_path,'diff_stamp_'+str(params['stamp_number'])+'.fits')
        image = fits.getdata(stamp)
        statistics.append([params['stamp_number'],np.median(image), image.std()])

    log.info('Calculated statistics on difference images')

    return statistics

def plot_dimage_statistics(params,dimage_stats,diff_images):

    markers = ['.', 'v', 's', 'p', '*', '+', 'X', 'd', '1', '3', 'D', '^', 'P', '>', '<', '4']
    col_keys = list(mcolors.TABLEAU_COLORS.keys()) + list(mcolors.TABLEAU_COLORS.keys())

    dimage_index = np.arange(0,len(diff_images),1)
    frames = []
    for f in diff_images:
        frames.append(path.basename(f))

    (fig, (ax0, ax1)) = plt.subplots(nrows=2, ncols=1)
    fig.set_size_inches(40, 10)

    if params['stamp_number'] == -1:
        stamp_index = np.arange(0,dimage_stats.shape[1],1)
    else:
        stamp_index = np.array([params['stamp_number']])

    for stamp_idx in range(0,dimage_stats.shape[1],1):
        ax0.plot(dimage_index, dimage_stats[:,stamp_idx,1], marker=markers[stamp_idx],
                    markerfacecolor=mcolors.TABLEAU_COLORS[col_keys[stamp_idx]],
                    markeredgecolor=mcolors.TABLEAU_COLORS[col_keys[stamp_idx]])
        ax0.set(xlabel='Image', ylabel='Mean pixel value [ADU]')

        ax1.plot(dimage_index, dimage_stats[:,stamp_idx,2], marker=markers[stamp_idx],
                    markerfacecolor=mcolors.TABLEAU_COLORS[col_keys[stamp_idx]],
                    markeredgecolor=mcolors.TABLEAU_COLORS[col_keys[stamp_idx]])
        ax1.set(xlabel='Image', ylabel='Std. Dev [ADU]')

    for ax in [ax0, ax1]:
        ax.grid()
        ax.set_xticks(dimage_index)
        ax.set_xticklabels(frames)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

    plt.tight_layout()
    plt.savefig(path.join(params['red_dir'], 'diff_image_statistics.png'), )

def mirror_mag_columns(photometry, phot_columns, log):

    (mag_col, merr_col) = plot_rms.get_photometry_columns(phot_columns)

    photometry[:,:,23] = photometry[:,:,mag_col]
    photometry[:,:,24] = photometry[:,:,merr_col]

    log.info('Mirrored '+phot_columns+' photometry data to corrected columns ready for post-processing')

    idx = np.where(photometry[:,:,13] > 0.0)
    mask = np.ma.getmask(photometry)

    return photometry

def set_image_photometry_qc_flags(photometry, log):

    mask = np.ma.getmask(photometry)
    idx = np.where(mask[:,:,13] == True)
    photometry[idx[0],idx[1],25] = -1

    log.info('Set quality control flag for datapoints from images with excessive photometric residuals')

    return photometry

def set_star_photometry_qc_flags(photometry, phot_stats, log):
    """Function to evaluate the photometric uncertainty of each datapoint, and
    flag datapoints with excessive uncertainties as suspect.
    The scaling relation used to make this determination was based on a fit to the
    weighted RMS .vs. weighted mean magnitude data for ROME-FIELD-01"""

    a0 = 0.232444
    a0_error = 0.0006723
    a1 = -5.0562
    a1_error = 0.01163
    rms = 0.310577

    max_uncertainty = 10**(a0 * phot_stats[:,0] + a1 + rms)

    error_threshold = np.zeros((photometry.shape[0], photometry.shape[1]))
    for i in range(0,photometry.shape[1],1):
        error_threshold[:,i] = max_uncertainty

    idx = np.where(photometry[:,:,24] > error_threshold)
    photometry[idx[0],idx[1],25] -= 1

    log.info('Set quality control flag for datapoints with photometric uncertainties exceeding mag-dependend threshold')

    return photometry

if __name__ == '__main__':

    (setup, params) = get_args()
    (status, report) = run_postproc(setup, **params)
