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
import photutils
from astropy.stats import SigmaClip
import numpy as np

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
    ref_exptime = reduction_metadata.headers_summary[1]['EXPKEY'][i][0]

    # Retrieve the photometric calibration parameters
    fit_params = [reduction_metadata.phot_calib[1]['a0'][0],
                  reduction_metadata.phot_calib[1]['a1'][0]]
    covar_fit = np.array([[reduction_metadata.phot_calib[1]['c0'][0], reduction_metadata.phot_calib[1]['c1'][0]],
                          [reduction_metadata.phot_calib[1]['c2'][0], reduction_metadata.phot_calib[1]['c3'][0]]
                          ])
    log.info('Calculating calibrated photometry using fit parameters: ' + repr(fit_params))
    log.info('and covarience matrix: ' + repr(covar_fit))

    # Extract the reference image detected sources catalog, and sort it to produce a list of
    # objects in order of flux:
    refcat = np.c_[
        reduction_metadata.star_catalog[1]['x'].data,
        reduction_metadata.star_catalog[1]['y'].data,
        reduction_metadata.star_catalog[1]['ref_flux'].data
    ]
    idx = refcat[:,2].argsort()[::-1]
    refcat = refcat[idx]

    # Loop over all selected images
    logs.ifverbose(log, setup, 'Performing aperture photometry for each image:')
    for image in new_images:
        logs.ifverbose(log, setup,
                       ' -> ' + os.path.basename(image))
        # Retrieve image pixel data and image parameters
        image_path = os.path.join(setup.red_dir, 'data', image)
        image_structure = image_handling.determine_image_struture(image_path, log=log)
        data_image = image_handling.get_science_image(image_path, image_structure=image_structure)

        i = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == os.path.basename(image))[0]
        aperture_radius = reduction_metadata.images_stats[1]['FWHM'][i][0]
        exptime = reduction_metadata.headers_summary[1]['EXPKEY'][i][0]

        # Perform object detection on the image
        detected_objects = starfind.detect_sources(setup, reduction_metadata,
                                                   image_path,
                                                   data_image,
                                                   log,
                                                   diagnostics=False)

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
                       ' -> alignment: ' + repr(align[0]))

        # Transform the positions of objects in the reference star_catalog to their corresponding positions in
        # the current image
        xx, yy, zz = np.dot(np.linalg.pinv(align[0].params),
                            np.r_[[refcat[:,0], refcat[:,1], [1] * len(refcat[:,2])]])
        transformed_star_positions = np.c_[xx, yy]

        # Perform aperture photometry at the transformed positions - for two apertures?
        residuals_xy = refcat[:,[0,1]] - transformed_star_positions
        phot_table, local_bkgd = ap_phot_image(data_image, transformed_star_positions, radius=aperture_radius)

        # Calculate the photometric scale factors for all stars in this image
        photscales = calc_phot_scale_factor(
            phot_table['aperture_sum'].value,
            refcat[:,2],
            exptime, ref_exptime
        )

        # Calculate fluxes normalized by the photometric scale factors:
        cal_flux = scale_photometry(
            phot_table['aperture_sum'].value,
            phot_table['aperture_sum_err'].value,
            photscales,
            exptime
        )

        # Convert the measured instrumental fluxes to magnitudes, scaled by the exposure time.
        # Note this applies to the instrumental measured fluxes,
        # but the calibrated fluxes have already been scaled
        (mag, mag_err, flux, flux_err) = photometry.convert_flux_to_mag(
            phot_table['aperture_sum'].value,
            phot_table['aperture_sum_err'].value,
            exp_time=exptime
        )
        (_, _, bkgd_flux, bkgd_flux_err) = photometry.convert_flux_to_mag(
            local_bkgd[:,0],
            local_bkgd[:,1],
            exp_time=exptime
        )

        (cal_mag, cal_mag_err, _, _) = photometry.convert_flux_to_mag(
            cal_flux[:,0],
            cal_flux_err[:,1],
            exp_time=None
        )

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
            new_image,                        # Image identifier
            log,
        )

        print(photometry_data)
        breakpoint()
        exit()

    # Store timeseries photometry
    status = 'OK'
    report = 'OK'
    log.info('Aperture photometry: ' + report)
    logs.close_log(log)

    return status, report

def ap_phot_image(data, pos, radius=3.0):
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

    return phot_table, local_bkgd


def calc_phot_scale_factor(flux, ref_flux, exptime, ref_exptime):
    """
    Function to calculate the photometric scale factor for all stars in
    a single image.

    Parameters:
        flux    array   Stellar fluxes in the working image
        ref_flux array  Stellar fluxes in the reference image
        exptime  float      Exposure time of the working image [s]
        ref_exptime float   Exposure time of the reference image [s]

    Returns:
        photscales 2D array Photometric scale factors and uncertainties
    """

    photscales = []

    # Mask requires that stars have valid measurements in both the
    # working image and the reference image
    mask = (~np.isnan(flux)) & (~np.isnan(ref_flux))

    # Photometric scale factor is calculated from the ratio of a star's flux in the
    # reference/working image, factored by the ratio of the exposure times.
    a = np.nanmedian(ref_flux[mask] / flux[mask] * exptime / ref_exptime)
    sig_a = np.nanmedian(np.abs(ref_flux[mask] / flux[mask] * exptime / ref_exptime - a))
    photscales.append([a, sig_a])

    return np.array(photscales)

def scale_photometry(flux, eflux, pscal, exptime):
    """
    Function to scale the measured star fluxes by the photometric scale factor.

    Parameters:
        flux    array   Stellar fluxes in the working image
        eflux   array   Uncertainties on stellar fluxes
        pscal   array   Photometric scale factor and uncertainty
        exptime float   Exposure time of working image [s]

    Returns:
        cal_flux    array   Calibrated fluxes
        cal_flux_err array  Uncertainties on the calibrated fluxes
    """

    cal_flux = np.zeros((len(flux),2))

    cal_flux[:,0] = pscal[:,0] * flux / exptime
    cal_flux[:,1] = np.sqrt(eflux**2 * pscal[:,0]**2 + flux**2 * pscal[:,1]**2) / exptime

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

    # Transfer the photometric measurements to the standard array format for timeseries photometry,
    # for consistency with the DIA channel of the pipeline, and consistent data products.
    # Note that this format contains residual indices of the photometry database - these are no longer used
    # Removing them is a task for v2.
    log.info('Starting to array data transfer')
    photometry_data[star_dataset_index,image_dataset_index,0] = star_field_ids
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
    photometry_data[star_dataset_index,image_dataset_index,17] = cal_flux
    photometry_data[star_dataset_index,image_dataset_index,18] = cal_flux_err
    photometry_data[star_dataset_index,image_dataset_index,19] = photscales[0]
    photometry_data[star_dataset_index,image_dataset_index,20] = photscales[1]
    photometry_data[star_dataset_index,image_dataset_index,21] = bkgd_flux
    photometry_data[star_dataset_index,image_dataset_index,22] = bkgd_flux_err

    log.info('Completed transfer of data to the photometry array')

    return photometry_data