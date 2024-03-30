import copy
import os
from astropy.io import fits
from pyDANDIA import logs
from pyDANDIA import metadata
from pyDANDIA import pipeline_setup
from pyDANDIA import photometry
from pyDANDIA import starfind
from pyDANDIA import image_handling
from pyDANDIA import stage4
from pyDANDIA import stage6
from pyDANDIA import hd5_utils
import photutils
from astropy.stats import SigmaClip
import numpy as np
import matplotlib.pyplot as plt

VERSION = 'pyDANDIA_ap_phot_v0.0.1'

def run_aperture_photometry(setup, **kwargs):
    """
    Driver function for aperture photometry, following on from stages 0-3.

    Args:
        setup: pipeline Setup object
        **kwargs:

    Returns:
        status: string Status of reduction stage
        report: string Description of reduction outcome

    The timeseries photometry is output to a pyDANDIA standard-format HDF5 file.
    The measured flux and uncertainties, and their corresponding magnitude columns contain the
    instrumental flux measurements.
    The calibrated flux and magnitude columns contain the measured fluxed scaled by the
    photometric scale factor.
    """

    log = logs.start_stage_log(setup.red_dir, 'aperture_photometry', version=VERSION)

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(metadata_directory=setup.red_dir,
                                         metadata_name='pyDANDIA_metadata.fits')

    # Identify the images that should be reduced from the metadata's reduction_status table
    all_images = reduction_metadata.find_all_images(setup, reduction_metadata,
                                                    os.path.join(setup.red_dir, 'data'), log=log)

    new_images = reduction_metadata.find_images_need_to_be_process(setup, all_images,
                                                                   stage_number=4, rerun_all=False, log=log)

    # Load or initialize the holding array for the timeseries photometry
    # This takes care of loading the data from any existing reduction
    photometry_data = stage6.build_photometry_array(
        setup,
        len(all_images),
        len(reduction_metadata.star_catalog[1]),
        log
    )

    # Retrieve the reference image pixel data and metadata:
    ref_image_path = os.path.join(str(reduction_metadata.data_architecture[1]['REF_PATH'][0]),
                               str(reduction_metadata.data_architecture[1]['REF_IMAGE'][0]))
    ref_structure = image_handling.determine_image_struture(ref_image_path, log=log)
    ref_image = image_handling.get_science_image(ref_image_path, image_structure=ref_structure)

    i = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == os.path.basename(ref_image_path))[0]
    ref_exptime = float(reduction_metadata.headers_summary[1]['EXPKEY'][i][0])

    # Retrieve the photometric calibration parameters
    fit_params = [reduction_metadata.phot_calib[1]['a0'][0],
                  reduction_metadata.phot_calib[1]['a1'][0]]
    covar_fit = np.array([[reduction_metadata.phot_calib[1]['c0'][0], reduction_metadata.phot_calib[1]['c1'][0]],
                          [reduction_metadata.phot_calib[1]['c2'][0], reduction_metadata.phot_calib[1]['c3'][0]]
                          ])
    log.info('Calculating calibrated photometry using fit parameters: ' + repr(fit_params))
    log.info('and covarience matrix: ' + repr(covar_fit))

    # Extract the reference image detected sources catalog, and sort it to produce a list of
    # objects in order of flux.  There are two copies of this array so that one can be
    # concenated into a single row for the dot operation later
    refcat = np.c_[
        reduction_metadata.star_catalog[1]['x'].data,
        reduction_metadata.star_catalog[1]['y'].data,
        reduction_metadata.star_catalog[1]['ref_flux'].data
    ]
    ref_objects = np.r_[[
        reduction_metadata.star_catalog[1]['x'].data,
        reduction_metadata.star_catalog[1]['y'].data,
        [1] * len(reduction_metadata.star_catalog[1])
    ]]
    star_order = refcat[:,2].argsort()[::-1]
    refcat = refcat[star_order]

    # To set the radius for aperture photometry, we use the PSF radius determined in stage 3:
    aperture_radius = reduction_metadata.psf_dimensions[1]['psf_radius'][0]

    # DEBUG
    star_idx = 10700

    # Loop over all selected images
    logs.ifverbose(log, setup, 'Performing aperture photometry for each image:')
    for k,image in enumerate(new_images):
        logs.ifverbose(log, setup,
                       ' -> ' + os.path.basename(image) + ', ' + str(k) + ' of ' + str(len(new_images)))

        # Retrieve image pixel data and image parameters
        image_path = os.path.join(setup.red_dir, 'data', image)
        image_structure = image_handling.determine_image_struture(image_path, log=log)
        data_image = image_handling.get_science_image(image_path, image_structure=image_structure)

        # Note thst by pipeline convention this index is taken from the headers_summary table.
        # Other tables in the metadata don't necessarily respect this ordering!
        image_idx = np.where(reduction_metadata.headers_summary[1]['IMAGES'] == os.path.basename(image))[0][0]
        #aperture_radius = reduction_metadata.images_stats[1]['FWHM'][i][0]
        exptime = float(reduction_metadata.headers_summary[1]['EXPKEY'][image_idx])
        print('Image is ' + str(image_idx) + ' in dataset index, with exptime = ' + str(exptime)+'s')

        # Perform object detection on the image
        detected_objects = starfind.detect_sources(setup, reduction_metadata,
                                                   image_path,
                                                   data_image,
                                                   log,
                                                   diagnostics=False)
        log.info(' --> Detected ' + str(len(detected_objects)) + ' stars in frame')

        # Sort the detected source catalogs from the working image
        datacat = np.c_[
            detected_objects['x'].data,
            detected_objects['y'].data,
            detected_objects['ref_flux'].data
        ]

        idx = datacat[:, 2].argsort()[::-1]
        datacat = datacat[idx]

        # Calculate the x, y offsets between the reference star_catalog and the objects in this frame
        align = stage4.find_init_transform(ref_image, data_image, refcat[:250], datacat[:250])
        logs.ifverbose(log, setup,
                       ' --> alignment: ' + repr(align[0]))

        # Transform the positions of objects in the reference star_catalog to their corresponding positions in
        # the current image.  Note that we cannot use the refcat here because it requires a 1D array in the
        # order of the reference image star catalog.
        xx, yy, zz = np.dot(np.linalg.pinv(align[0].params),ref_objects)
        transformed_star_positions = np.c_[xx, yy]

        # Perform aperture photometry at the transformed positions - for two apertures?
        residuals_xy = refcat[:,[0,1]] - transformed_star_positions
        phot_table, local_bkgd = ap_phot_image(data_image, transformed_star_positions,
                                               radius=aperture_radius, log=log)

        print('RAW: ', phot_table[star_idx])
        # Convert the measured instrumental fluxes to magnitudes, scaled by the exposure time.
        # Note this applies to the instrumental measured fluxes,
        # but the calibrated fluxes have already been scaled
        (mag, mag_err, flux, flux_err) = photometry.convert_flux_to_mag(
            phot_table['aperture_sum'].value,
            phot_table['aperture_sum_err'].value,
            exp_time=exptime
        )
        (_, _, bkgd_flux, bkgd_flux_err) = photometry.convert_flux_to_mag(
            local_bkgd['aperture_sum'].value,
            local_bkgd['aperture_sum_err'].value,
            exp_time=exptime
        )
        print('INST: ',flux[star_idx], flux_err[star_idx], mag[star_idx], mag_err[star_idx])

        # Calculate the photometric scale factors for all stars in this image
        photscales = calc_phot_scale_factor(
            setup,
            image,
            flux,
            reduction_metadata.star_catalog[1]['ref_flux'].data,    # Already scaled by exposure time
            exptime, ref_exptime, log=log, diagnostics=True
        )

        # Calculate fluxes normalized by the photometric scale factors:
        cal_flux = scale_photometry(
            flux,
            flux_err,
            photscales,
            log=log
        )

        (cal_mag, cal_mag_err, _, _) = photometry.convert_flux_to_mag(
            cal_flux[:,0],
            cal_flux[:,1],
            exp_time=None
        )
        print('CAL: ',cal_flux[star_idx,0], cal_flux[star_idx,1], cal_mag[star_idx], cal_mag_err[star_idx])

        # Store the photometry from this image to the main array
        photometry_data = store_stamp_photometry_to_array(
            reduction_metadata,
            photometry_data,
            residuals_xy,
            flux, flux_err,                   # Exposure-scaled raw flux measurements
            bkgd_flux, bkgd_flux_err,         # Exposure-scaled background flux measurements
            photscales,                       # Photometric scale factors
            cal_flux,                         # PS-scaled fluxes
            mag, mag_err,                     # Exposure-scaled raw flux measurements in mag
            cal_mag, cal_mag_err,             # PS-scaled fluxes in mag
            aperture_radius,                  # Aperture radius used for photometry
            image,                            # Image identifier
            log,
        )
        print('PHOTO: ', photometry_data[star_idx, image_idx, :])

    # Update the metadata reduction status
    reduction_metadata.update_reduction_metadata_reduction_status(new_images, stage_number=4, status=1, log=log)
    reduction_metadata.update_reduction_metadata_reduction_status(new_images, stage_number=5, status=1, log=log)
    reduction_metadata.update_reduction_metadata_reduction_status(new_images, stage_number=6, status=1, log=log)
    reduction_metadata.software[1]['stage6_version'] = VERSION
    reduction_metadata.save_updated_metadata(
        reduction_metadata.data_architecture[1]['OUTPUT_DIRECTORY'][0],
        reduction_metadata.data_architecture[1]['METADATA_NAME'][0],
        log=log)

    # Store timeseries photometry
    hd5_utils.write_phot_hd5(setup, photometry_data, log=log)

    status = 'OK'
    report = 'OK'
    log.info('Aperture photometry: ' + report)
    logs.close_log(log)

    return status, report

def ap_phot_image(data, pos, radius=3.0, log=None):
    """
    Function to perform aperture photometry on a single image, given an input source catalog.
    Based on a function by Etienne Bachelet

    Args:
        image: numpy image pixel data 2D array
        positions: 2D numpy array of star pixel positions (x,y)
        radius: float radius of the aperture to use for photometry in pixels

    Returns:
        phot_table: photutils output table
    """

    rad = 2*radius
    apertures = photutils.CircularAperture(pos, r=rad)
    sigma_clip = SigmaClip(sigma=3.)
    bkg_estimator = photutils.MedianBackground()
    bkg = photutils.Background2D(
        data, (50, 50),
        filter_size=(3, 3),
        sigma_clip=sigma_clip,
        bkg_estimator=bkg_estimator)
    #breakpoint()
    ron = 0
    gain = 1

    # Perform photometry of all stars on the image data
    error = photutils.utils.calc_total_error(np.abs(data), (bkg.background_rms**2)**0.5, gain)
    error = (error**2+ron**2/gain**2)**0.5
    phot_table = photutils.aperture_photometry(data-bkg.background, apertures, method='subpixel',error=error)

    # Estimate the background flux at the position of all stars from the background, using the same aperturers
    bkg_error = photutils.utils.calc_total_error(np.abs(bkg.background), (bkg.background_rms**2)**0.5, gain)
    error = (bkg_error**2+ron**2/gain**2)**0.5
    local_bkgd = photutils.aperture_photometry(bkg.background, apertures, method='subpixel', error=error)

    if log:
        log.info(' --> Completed aperture photometry')

    return phot_table, local_bkgd


def calc_phot_scale_factor(setup, image, flux, ref_flux, exptime, ref_exptime, log=None, diagnostics=False):
    """
    Function to calculate the photometric scale factor for all stars in
    a single image.

    Parameters:
        setup   Setup obj for the reduction
        imaeg string  Filename of the working image
        flux    array   Stellar fluxes in the working image
        ref_flux array  Stellar fluxes in the reference image
        exptime  float      Exposure time of the working image [s]
        ref_exptime float   Exposure time of the reference image [s]

    Returns:
        photscales 2D array Photometric scale factors and uncertainties
    """

    # Mask requires that stars have valid measurements in both the
    # working image and the reference image
    mask = (~np.isnan(flux)) & (~np.isnan(ref_flux)) & (flux > 0.0) & (ref_flux > 0.0)

    # Photometric scale factor is calculated from the ratio of a star's flux in the
    # reference/working image, NOT factored by the ratio of the exposure times
    # because both ref_flux and flux have already been scaled to the flux in a 1s exposure.
    # There is typically a long tail to this distribution, caused by variables
    # in the field of view, so we exclude the extreme tail.
    #ratios = (ref_flux[mask] / flux[mask] * exptime / ref_exptime)
    for j in range(10700,10701,1):
        if mask[j]:
            print(flux[j], ref_flux[j])

    ratios = (ref_flux[mask] / flux[mask])
    rmin = np.percentile(ratios, 25.0)
    rmax = np.percentile(ratios, 50.0)
    rmin = 0.6
    rmax = 1.5
    print('R Range = ', rmin, rmax)
    selection = (ratios <= rmax) & (ratios >= rmin)
    a = np.nanmedian(ratios[selection])
    sig_a = np.nanmedian(np.abs(ratios[selection] - a))
    
    photscales = [a, sig_a]

    if log:
        log.info(' --> Photometric scale factor: ' + repr(photscales))
    print(' --> Photometric scale factor: ' + repr(photscales))
    print(ratios)
    print(rmax)
    if diagnostics:
        if not os.path.isdir(os.path.join(setup.red_dir, 'apphot')):
            os.mkdir(os.path.join(setup.red_dir, 'apphot'))
        fig = plt.figure(1, (10, 10))
        plt.hist(ratios, bins=100, range=(0,rmax*1.01))
        print('HERE 1')
        plt.xlabel('PS ratios per star')
        plt.ylabel('Count')
        (xmin, xmax, ymin, ymax) = plt.axis()
        plt.axis([0, rmax*1.01, ymin, ymax])
        plt.plot([a,a], [ymin,ymax], 'k-')
        print('HERE 2')
        plt.plot([a-sig_a]*2, [ymin,ymax], 'k-.')
        plt.plot([a+sig_a]*2, [ymin,ymax], 'k-.')
        print('HERE 3')
        plt.plot([rmin,rmin], [ymin,ymax], 'r-')
        plt.plot([rmax,rmax], [ymin,ymax], 'r-')
        print('HERE 4')
        plt.savefig(os.path.join(setup.red_dir, 'apphot', 'ps_ratios_hist_'+image.replace('.fits','')+'.png'))
        print('HERE 5')
        plt.close(1)

    return np.array(photscales)

def scale_photometry(flux, eflux, pscal, log=None):
    """
    Function to scale the measured star fluxes by the photometric scale factor.

    Parameters:
        flux    array   Stellar fluxes in the working image, scaled by exposure time
        eflux   array   Uncertainties on stellar fluxes
        pscal   array   Photometric scale factor and uncertainty

    Returns:
        cal_flux    array   Calibrated fluxes
        cal_flux_err array  Uncertainties on the calibrated fluxes
    """

    cal_flux = np.zeros((len(flux),2))

    cal_flux[:,0] = pscal[0] * flux
    cal_flux[:,1] = np.sqrt(eflux**2 * pscal[0]**2 + flux**2 * pscal[1]**2)

    if log:
        log.info(' --> normalized measured star fluxes by the photometric scale factor')

    return cal_flux

def store_stamp_photometry_to_array(
            reduction_metadata,
            photometry_data,
            residuals_xy,
            flux, flux_err,                   # Exposure-scaled raw flux measurements
            bkgd_flux, bkgd_flux_err,         # Exposure-scaled background flux measurements
            photscales,                       # Photometric scale factors
            cal_flux,                         # PS-scaled fluxes
            mag, mag_err,                     # Exposure-scaled raw flux measurements in mag
            cal_mag, cal_mag_err,             # PS-scaled fluxes in mag
            aperture_radius,                  # Aperture radius used for photometry
            new_image,                        # Image identifier
            log,
        ):
    """Function to store photometry data from a stamp to the main
    photometry array"""

    log.info('Starting to store photometry for image '+new_image)

    # The index of the data from a given image corresponds to the index of that
    # image in the metadata
    image_dataset_index = np.where(new_image == reduction_metadata.headers_summary[1]['IMAGES'].data)[0][0]
    hjd = reduction_metadata.headers_summary[1]['HJD'].data[image_dataset_index]
    star_dataset_ids = reduction_metadata.star_catalog[1]['index'].data
    star_dataset_ids = star_dataset_ids.astype('float')
    star_dataset_ids = star_dataset_ids.astype('int')
    star_dataset_index = star_dataset_ids - 1

    print('Storing photometry in image index ', image_dataset_index, ' star index ', star_dataset_index[10701])

    # Transfer the photometric measurements to the standard array format for timeseries photometry,
    # for consistency with the DIA channel of the pipeline, and consistent data products.
    # Note that this format contains residual indices of the photometry database - these are no longer used
    # Removing them is a task for v2.
    log.info('Starting to array data transfer')
    photometry_data[star_dataset_index,image_dataset_index,0] = star_dataset_ids
    photometry_data[star_dataset_index,image_dataset_index,1] = 0   # DB pk for reference image
    photometry_data[star_dataset_index,image_dataset_index,2] = 0   # DB pk for image
    photometry_data[star_dataset_index,image_dataset_index,3] = 0   # DB pk for stamp
    photometry_data[star_dataset_index,image_dataset_index,4] = 0   # DB pk for facility
    photometry_data[star_dataset_index,image_dataset_index,5] = 0   # DB pk for filter
    photometry_data[star_dataset_index,image_dataset_index,6] = 0   # DB pk for code
    photometry_data[star_dataset_index,image_dataset_index,7] = residuals_xy[:,0]   # Offset of stars from ref
    photometry_data[star_dataset_index,image_dataset_index,8] = residuals_xy[:,1]
    photometry_data[star_dataset_index,image_dataset_index,9] = hjd
    photometry_data[star_dataset_index,image_dataset_index,10] = aperture_radius
    photometry_data[star_dataset_index,image_dataset_index,11] = mag
    photometry_data[star_dataset_index,image_dataset_index,12] = mag_err
    photometry_data[star_dataset_index,image_dataset_index,13] = cal_mag
    photometry_data[star_dataset_index,image_dataset_index,14] = cal_mag_err
    photometry_data[star_dataset_index,image_dataset_index,15] = flux
    photometry_data[star_dataset_index,image_dataset_index,16] = flux_err
    photometry_data[star_dataset_index,image_dataset_index,17] = cal_flux[:,0]
    photometry_data[star_dataset_index,image_dataset_index,18] = cal_flux[:,1]
    photometry_data[star_dataset_index,image_dataset_index,19] = [photscales[0]] * len(star_dataset_index)
    photometry_data[star_dataset_index,image_dataset_index,20] = [photscales[1]] * len(star_dataset_index)
    photometry_data[star_dataset_index,image_dataset_index,21] = bkgd_flux
    photometry_data[star_dataset_index,image_dataset_index,22] = bkgd_flux_err

    print('OUT: ', flux[10700], flux_err[10700])
    print('OUT: ', photometry_data[10700,image_dataset_index,15])

    log.info('Completed transfer of data to the photometry array')

    return photometry_data