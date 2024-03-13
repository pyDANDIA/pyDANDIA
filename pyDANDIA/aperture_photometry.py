import os
from astropy.io import fits
from pyDANDIA import logs
from pyDANDIA import metadata
from pyDANDIA import pipeline_setup
from pyDANDIA import starfind
from pyDANDIA import image_handling
from pyDANDIA import stage4
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

    # Retrieve the reference image pixel data:
    ref_image_path = os.path.join(str(reduction_metadata.data_architecture[1]['REF_PATH'][0]),
                               str(reduction_metadata.data_architecture[1]['REF_IMAGE'][0]))
    ref_structure = image_handling.determine_image_struture(ref_image_path, log=log)
    ref_image = image_handling.get_science_image(ref_image_path, image_structure=ref_structure)

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
    times = []
    fluxes = []
    efluxes = []
    fluxes2 = []
    efluxes2 = []
    exptime = []
    fwhms = []
    for image in new_images:
        logs.ifverbose(log, setup,
                       ' -> ' + os.path.basename(image))
        # Retrieve image pixel data and image parameters
        image_path = os.path.join(setup.red_dir, 'data', image)
        image_structure = image_handling.determine_image_struture(image_path, log=log)
        data_image = image_handling.get_science_image(image_path, image_structure=image_structure)

        i = np.where(reduction_metadata.images_stats[1]['IM_NAME'] == os.path.basename(image))[0]
        data_fwhm = reduction_metadata.images_stats[1]['FWHM'][i][0]

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
        skip = False
        if not skip:
            xx, yy, zz = np.dot(np.linalg.pinv(align[0].params),
                                np.r_[[refcat[:,0], refcat[:,1], [1] * len(refcat[:,2])]])

            # Perform aperture photometry at the transformed positions - for two apertures?
            phot_table = ap_phot_image(data_image, np.c_[xx, yy], radius=data_fwhm)
            phot_table2 = ap_phot_image(data_image, np.c_[xx, yy], radius=3)
            print(phot_table)

            fluxes.append(phot_table['aperture_sum'].value)
            efluxes.append(phot_table['aperture_sum_err'].value)
            print(fluxes)

            fluxes2.append(phot_table2['aperture_sum'].value)
            efluxes2.append(phot_table2['aperture_sum_err'].value)

            # NEED HJDS BY THIS POINT
            i = np.where(reduction_metadata.headers_summary[1]['IMAGES'] == os.path.basename(image))[0]
            times.append(reduction_metadata.headers_summary[1]['HJD'][i][0])
            exptime.append(reduction_metadata.headers_summary[1]['EXPKEY'][i][0])
            fwhms.append(data_fwhm)

    # Scale the photometry by the image exposure time
    #pscale = phot_scales(fluxes, exptime, refind=0, sub_catalog=np.arange(100, 200))
    #pscale2 = phot_scales(fluxes2, exptime, refind=0, sub_catalog=np.arange(100, 200))
    #phot = final_phot(times, fluxes, efluxes, pscale, exptime)
    #phot2 = final_phot(times, fluxes2, efluxes2, pscale2, exptime)

    # Store timeseries photometry
    status = 'OK'
    report = 'OK'
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