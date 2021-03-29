from os import path, rename
from sys import argv
import numpy as np
from pyDANDIA import crossmatch
from pyDANDIA import hd5_utils
from pyDANDIA import logs
from pyDANDIA import metadata
from pyDANDIA import plot_rms
from pyDANDIA import pipeline_setup
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from astropy import visualization

def run_postproc():
    """Driver function for post-processing:
    Assessment of photometric residuals and uncertainties
    """
    params = get_args()

    log = logs.start_stage_log( params['log_dir'], 'postproc_phot' )

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(params['red_dir'], 'pyDANDIA_metadata.fits')
    phot_file = path.join(params['red_dir'],'photometry.hdf5')
    photometry = hd5_utils.read_phot_from_hd5_file(phot_file, return_type='array')
    log.info('Loaded dataset photometry and metadata')

    # Grow photometry array to allow additional columns for corrected mags
    photometry = grow_photometry_array(photometry,log)
    photometry = mask_photometry_array(reduction_metadata, photometry, log)

    # Calculate mean_mag, RMS for all stars
    phot_stats = plot_rms.calc_mean_rms_mag(photometry,log,'calibrated')
    plot_rms.plot_rms(phot_stats, params, log,
                    plot_file=path.join(params['red_dir'],'init_rms_mag.png'))
    plot_rms.output_phot_statistics(phot_stats,
                                    path.join(params['red_dir'],'init_rms_mag.txt'),
                                    log)

    # Calculate photometric residuals
    phot_residuals = calc_phot_residuals(photometry, phot_stats, log, 'calibrated')
    plot_phot_residuals(params, reduction_metadata, phot_residuals, log)

    # Calculate mean residual per image
    image_residuals = calc_image_residuals(reduction_metadata, photometry, phot_residuals,log)
    plot_image_residuals(params, image_residuals, log)

    # Mask photometry from images with excessive average photometric residuals
    photometry = mask_phot_from_bad_images(photometry, image_residuals, log)

    # Mirror calibrated photometry to corrected columns ready for processing:
    photometry = mirror_mag_columns(photometry, 'calibrated', log)

    # Compensate for mean residual per image and output to photometry array
    #photometry = apply_image_mag_correction(params, image_residuals, photometry, log, 'calibrated')

    # Re-calculate mean_mag, RMS for all stars, using corrected magnitudes
    print('Recalculating statistics')
    phot_stats = plot_rms.calc_mean_rms_mag(photometry,log,'corrected')
    phot_stats = mask_photometry_stats(phot_stats, log)
    plot_rms.plot_rms(phot_stats, params, log,
                    plot_file=path.join(params['red_dir'],'postproc_rms_mag.png'))
    plot_rms.output_phot_statistics(phot_stats,
                path.join(params['red_dir'],'postproc_rms_mag.txt'),
                log)

    # Re-calculate photometric residuals
    phot_residuals = calc_phot_residuals(photometry, phot_stats, log, 'corrected')

    # Calculate photometric scatter per image
    image_rms = calc_image_rms(phot_residuals, log)

    # Factor photometric scatter into photometric residuals
    photometry = apply_image_merr_correction(photometry, image_rms, log, 'corrected')

    # Set quality control flags to indicate suspect data points in photometry array
    photometry = set_photometry_qc_flags(photometry, log)

    # Ouput updated photometry
    output_revised_photometry(params, photometry, log)

    log.info('Post-processing: complete')

    logs.close_log(log)

def get_args():

    params = {}
    if len(argv) == 1:
        params['red_dir'] = input('Please enter the path to the datasets reduction directory: ')
        opt = input('Generate diagnostic plots?  Y or N: ')
    else:
        params['red_dir'] = argv[1]
        if len(argv) == 3:
            opt = argv[2]
        else:
            opt = 'N'

    if 'y' in str(opt).lower():
        params['diagnostic_plots'] = True
    else:
        params['diagnostic_plots'] = False

    params['log_dir'] = params['red_dir']

    return params

def grow_photometry_array(photometry,log):

    (mag_col, merr_col) = plot_rms.get_photometry_columns('calibrated')

    if photometry.shape[2] < 26:
        new_photometry = np.zeros((photometry.shape[0], photometry.shape[1], photometry.shape[2]+3))
        new_photometry[:,:,0:photometry.shape[2]] = photometry
        log.info('Added three columns to the photometry array')
        return new_photometry
    else:
        log.info('Photometry array already has all 26 columns')
        return photometry

def mask_photometry_array(reduction_metadata, photometry, log):

    (mag_col, merr_col) = plot_rms.get_photometry_columns('calibrated')

    mask = np.invert(photometry[:,:,mag_col] > 0.0)

    expand_mask = np.empty((mask.shape[0], mask.shape[1], photometry.shape[2]))
    for col in range(0,expand_mask.shape[2],1):
        expand_mask[:,:,col] = mask

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

    test_star_idxs = [30001, 78283, 109708, 120495, 166501, 205177]
    init_lcs = []
    for star in test_star_idxs:
        init_lcs.append( photometry[star,:,[9,mag_col,merr_col]].T )

    photometry[:,:,photometry.shape[2]-2] = photometry[:,:,mag_col] - image_residuals[:,0]
    photometry[:,:,photometry.shape[2]-1] = np.sqrt( photometry[:,:,merr_col]*photometry[:,:,merr_col] + \
                                                        image_residuals[:,1]*image_residuals[:,1] )

    post_lcs = []
    for star in test_star_idxs:
        post_lcs.append( photometry[star,:,[9,photometry.shape[2]-2,photometry.shape[2]-1]].T )

    test_plot_lcs(params, test_star_idxs, init_lcs, post_lcs)

    log.info('Applied magnitude offset to calculate corrected magnitudes')

    return photometry

def test_plot_lcs(params, star_idxs, init_lcs, post_lcs):

    for j, star in enumerate(star_idxs):
        init_lc = init_lcs[j]
        post_lc = post_lcs[j]

        fig = plt.figure(1,(10,10))
        mean_mag = init_lc[:,1].mean()
        ymin2 = mean_mag + 1.0
        ymax2 = mean_mag - 1.0
        with_errors = False
        if with_errors:
            plt.errorbar(init_lc[:,0]-2450000.0,init_lc[:,1],
                    yerr=init_lc[:,2],
                    color='k', fmt='none')
            plt.errorbar(post_lc[:,0]-2450000.0,post_lc[:,1],
                    yerr=post_lc[:,2],
                    color='m', fmt='none')
        else:
            plt.plot(init_lc[:,0]-2450000.0,init_lc[:,1],'k.')
            plt.plot(post_lc[:,0]-2450000.0,post_lc[:,1],'m.')

        plt.xlabel('HJD-2450000.0')
        plt.ylabel('Mag')
        plt.title('Star '+str(star+1))
        (xmin,xmax,ymin1,ymax1) = plt.axis()
        plt.axis([xmin,xmax,ymin2,ymax2])
        plt.savefig(path.join(params['red_dir'],'test_lightcurve_'+str(star+1)+'.png'))
        plt.close(1)

def calc_image_rms(phot_residuals, log):

    err_squared_inv = 1.0 / (phot_residuals[:,:,1]*phot_residuals[:,:,1])
    rms =  np.sqrt( (phot_residuals[:,:,0]**2 * err_squared_inv).sum(axis=0) / (err_squared_inv.sum(axis=0)) )

    log.info('Calculated RMS of photometric residuals per image')

    return rms

def apply_image_merr_correction(photometry, image_rms, log, phot_columns):

    (mag_col, merr_col) = plot_rms.get_photometry_columns(phot_columns)

    photometry[:,:,photometry.shape[2]-1] = np.sqrt( photometry[:,:,merr_col]*photometry[:,:,merr_col] + \
                                                        image_rms*image_rms )

    log.info('Factored RMS of image residuals into photometric uncertainties')

    return photometry

def output_revised_photometry(params, photometry, log):

    # Back up the older photometry file for now
    phot_file_name = path.join(params['red_dir'],'photometry.hdf5')
    bkup_file_name = path.join(params['red_dir'],'photometry_stage6.hdf5')
    if path.isfile(phot_file_name):
        rename(phot_file_name, bkup_file_name)

    # Output file with additional columns:
    setup = pipeline_setup.pipeline_setup(params)
    hd5_utils.write_phot_hd5(setup,photometry,log=log)

def mask_phot_from_bad_images(photometry, image_residuals, log):

    residuals_threshold = 0.15
    idx = np.where(abs(image_residuals[:,0]) > residuals_threshold)[0]

    expand_mask = np.ma.getmask(photometry)
    for i in idx:
        mask = np.empty((photometry.shape[0],photometry.shape[2]), dtype='bool')
        mask.fill(True)
        expand_mask[:,i,:] = mask

    photometry = np.ma.masked_array(photometry[:,:,:], mask=expand_mask)

    log.info('Masked photometric data for images with excessive average residuals')

    return photometry

def mirror_mag_columns(photometry, phot_columns, log):

    (mag_col, merr_col) = plot_rms.get_photometry_columns(phot_columns)

    photometry[:,:,23] = photometry[:,:,mag_col]
    photometry[:,:,24] = photometry[:,:,merr_col]

    log.info('Mirrored '+phot_columns+' photometry data to corrected columns ready for post-processing')

    idx = np.where(photometry[:,:,23] > 0.0)
    print('INDEX: ',idx)
    idx = np.where(photometry[:,:,13] > 0.0)
    print('INDEX: ',idx)
    mask = np.ma.getmask(photometry)
    print(mask[:,:,23])
    print((mask[:,:,23] == True).all())
    return photometry

def set_photometry_qc_flags(photometry, log):

    mask = np.ma.getmask(photometry)
    idx = np.where(mask[:,:,13] == True)
    photometry[idx[0],idx[1],25] = -1

    log.info('Set quality control flag for datapoints with excessive photometric residuals')

    return photometry

if __name__ == '__main__':
    run_postproc()
