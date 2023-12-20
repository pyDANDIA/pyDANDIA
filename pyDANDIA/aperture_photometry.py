import os
from astropy.io import fits
from pyDANDIA import  logs
from pyDANDIA import  metadata
from pyDANDIA import  pipeline_setup
from pyDANDIA import  starfind
import photutils
from astropy.stats import SigmaClip
import numpy as np

VERSION = 'pyDANDIA_ap_phot_v0.0.1'

def run_aperture_photometry(setup, **kwargs):
    """
    Driver function for aperture photometry

    Args:
        setup: pipeline Setup object
        **kwargs:

    Returns:
        status: string Status of reduction stage
        report: string Description of reduction outcome
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

    # Loop over all selected images
    for image in new_images:
        image_path = os.path.join(setup.red_dir, 'data', image)

        # Perform object detection on the image
        (status, report, params) = starfind.starfind(setup, image_path, reduction_metadata,
                                                     plot_it=False, log=log, thumbsize=500)

        # Calculate the x, y offsets between the reference star_catalog and the objects in this frame
        align = stage4.find_init_transform(ref[1].data, data[1].data,
                                           refcat[:250],
                                           datacat[:250])

        # Transform the positions of objects in the reference star_catalog to their corresponding positions in
        # the current image
        xx, yy, zz = np.dot(np.linalg.pinv(align[0]),
                            np.r_[[ref_cat['xcentroid'], ref_cat['ycentroid'], [1] * len(ref_cat['flux'])]])


        # Perform aperture photometry at the transformed positions - for two apertures?
        phot_table = ap_phot_image(data[1].data, np.c_[xx, yy], radius=data_fwhm)
        phot_table2 = ap_phot_image(data[1].data, np.c_[xx, yy], radius=3)

        fluxes.append(phot_table['aperture_sum'].value)
        efluxes.append(phot_table['aperture_sum_err'].value)

        fluxes2.append(phot_table2['aperture_sum'].value)
        efluxes2.append(phot_table2['aperture_sum_err'].value)

        times.append(data[1].header['MJD-OBS'])
        exptime.append(data[1].header['EXPTIME'])
        fwhms.append(data_fwhm)

    # Scale the photometry by the image exposure time
    pscale = phot_scales(fluxes, exptime, refind=0, sub_catalog=np.arange(100, 200))
    pscale2 = phot_scales(fluxes2, exptime, refind=0, sub_catalog=np.arange(100, 200))
    phot = final_phot(times, fluxes, efluxes, pscale, exptime)
    phot2 = final_phot(times, fluxes2, efluxes2, pscale2, exptime)

    # Store timeseries photometry

    log.info('Aperture photometry: ' + report)
    logs.close_log(log)

    return status, report

def ap_phot_image(data,pos,radius=3.0):
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
    bkg = photutils.Background2D(data, (50, 50),  filter_size=(3, 3), sigma_clip=sigma_clip, bkg_estimator=bkg_estimator)
    #breakpoint()
    ron = 0
    gain = 1

    error = photutils.utils.calc_total_error(np.abs(data), (bkg.background_rms**2)**0.5, gain)
    error = (error**2+ron**2/gain**2)**0.5
    phot_table = photutils.aperture_photometry(data-bkg.background, apertures, method='subpixel',error=error)

    return phot_table