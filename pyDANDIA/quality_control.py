# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 13:34:31 2018

@author: rstreet
"""
import os
from astropy.io import fits
from astropy import units as u
from astropy.coordinates import SkyCoord
import numpy as np
from pyDANDIA import metadata

def verify_stage0_output(setup,log):
    """Function to verify that stage 0 has produced the expected output.

    This function checks for the presences of a metadata file and a stage log
    """

    log.info('Verifying stage 0 data products:')

    status = 'OK'
    report = 'Completed successfully'

    metadata_file = os.path.join(setup.red_dir, 'pyDANDIA_metadata.fits')
    stage_log = os.path.join(setup.red_dir, 'stage0.log')

    if os.path.isfile(metadata_file) == False or os.path.isfile(stage_log) == False:

        status = 'ERROR'
        report = 'Stage 0 finished without producing its expected data products'

        log.info('Status: '+status)
        log.info('Report: '+report)

        return status, report

    m = fits.open(metadata_file)

    if len(m) < 6:

        status = 'ERROR'
        report = 'Stage 0 produced an incomplete metadata file'

        log.info('Status: '+status)
        log.info('Report: '+report)

        return status, report

    log.info('Status: '+status)
    log.info('Report: '+report)

    return status, report

def assess_image(reduction_metadata,image_params,image_header,log):
    """Function to assess the quality of an image, and flag any which should
    be considered suspect.  Flagged images, particularly those taken in
    conditions of poor seeing, can dramatically affect the reduction timescale.

    Inputs:
        :param Metadata reduction_metadata: Metadata for the reduction, including
                                            the reduction parameters table
        :param dict image_params: Measured parameters of the image including
                                  the FWHM, sky background and number of stars

    Outputs:
        :param int use_phot: Quality flag whether this image can be used for photometry
        :param int use_ref: Quality flag whether this image could be used for a reference
        :param int use_image: Quality flag whether this image should be reduced at all

    Flag convention: 1 = OK, 0 = bad image
    """

    use_phot = 1
    use_ref = 1
    use_image = 1
    report = ''

    sigma_max = reduction_metadata.reduction_parameters[1]['MAX_SIGMA_PIXELS'][0]
    sky_max = reduction_metadata.reduction_parameters[1]['MAX_SKY'][0]
    sky_ref_max = reduction_metadata.reduction_parameters[1]['MAX_SKY_REF'][0]

    if image_params['nstars'] == 0:
        use_phot = 0
        use_ref = 0
        use_image = 0
        report = append_errors(report, 'No stars detected in frame')

    if image_params['sigma_x'] > sigma_max or image_params['sigma_y'] > sigma_max:
        use_phot = 0
        use_ref = 0
        report = append_errors(report, 'FWHM exceeds threshold')

    if image_params['sky'] > sky_max:
        use_phot = 0
        report = append_errors(report, 'Sky background exceeds threshold for photometry')

    if image_params['sky'] > sky_ref_max:
        use_ref = 0
        report = append_errors(report, 'Sky background exceeds threshold for reference')

    if not verify_telescope_pointing(image_header):
        use_image = 0
        use_phot = 0
        use_ref = 0
        report = append_errors(report, 'Telescope pointing error exceeds threshold')

    if use_phot == 1 and use_ref == 1 and use_image == 1 and len(report) == 0:
        report = 'OK'

    log.info('Quality assessment:')
    log.info('Use for photometry = '+str(use_phot))
    log.info('Use for reference = '+str(use_ref))
    log.info('Reduce image at all = '+str(use_image))
    log.info('Report: '+report)

    return use_phot, use_ref, use_image, report

def append_errors(report,error):

    if len(report) == 0:

        report = error

    else:

        report += ', '+error

    return error


def verify_stage1_output(setup,log):
    """Function to verify that stage 0 has produced the expected output.

    This function checks for the presences of a metadata file and a stage log
    """

    log.info('Verifying stage 1 data products:')

    status = 'OK'
    report = 'Completed successfully'

    stage_log = os.path.join(setup.red_dir, 'stage1.log')

    if os.path.isfile(stage_log) == False:

        status = 'ERROR'
        report = 'Stage 0 finished without producing its stage log'

        log.info('Status: '+status)
        log.info('Report: '+report)

        return status, report

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'images_stats' )

    image_stats = reduction_metadata.images_stats[1]

    for flag in image_stats['USE_PHOT'].data:

        if flag != 0 and flag != 1:

            status = 'ERROR'
            report = 'Stage 1 produced unrecognised values in the metadata image stats table'

            log.info('Status: '+status)
            log.info('Report: '+report)

            return status, report

    log.info('Status: '+status)
    log.info('Report: '+report)

    return status, report

def verify_telescope_pointing(image_header):
    """Function to compare the CAT-RA, CAT-DEC requested for an observation with the
    RA, DEC of the telescopes pointing.  Images with a discrepancy greater than 1 arcmin
    are flagged as bad

    Input:
        :param image_header FITS image header object

    Output:
        :param image_status Boolean: Flag indicating whether the image passes
                                     the QC test
    """

    threshold = (1.0/60.0) * u.deg

    requested_pointing = SkyCoord(image_header['CAT-RA']+' '+image_header['CAT-DEC'],
                                  frame='icrs', unit=(u.hourangle, u.deg))

    actual_pointing = SkyCoord(image_header['RA']+' '+image_header['DEC'],
                                  frame='icrs', unit=(u.hourangle, u.deg))

    if requested_pointing.separation(actual_pointing) <= threshold:
        return True
    else:
        return False

def verify_image_shifts(new_images, shift_data, image_red_status):
    """Function to review the measured pixel offsets of each image from the
    reference for that dataset, and ensure that any severely offset images
    are marked as bad.  These images are removed from the new_images list.

    Inputs:
        :param list new_images: list of image names to process
        :param list shift_data: list of measured image shifts
        :param dict image_red_status: Reduction status of each image for the
                                      current reduction stage
    Outputs:
        :param dict image_red_status:
    """

    threshold = 100.0 # pixels

    for i,entry in enumerate(shift_data):
        image_list = np.array(new_images)
        image = entry[0]
        if abs(entry[1]) >= threshold or abs(entry[2]) >= threshold:
            image_red_status[image] = '-1'

            idx = np.where(image_list == image)
            if len(idx[0]) > 0:
                rm_image = new_images.pop(idx[0][0])

    return new_images, image_red_status
