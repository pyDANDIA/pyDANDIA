######################################################################
#
# stage2.py - Second stage of the pipeline.
#             Reads metadata and selects a reference image
#
# dependencies:
#      numpy 1.8+
#      astropy 1.0+
######################################################################

import numpy as np
import os
from astropy.io import fits
import sys

import config_utils
from astropy.table import Table
from astropy.nddata import Cutout2D
from operator import itemgetter

import metadata
import pixelmasks
import logs
import stage0


def run_stage2(setup):
    """Main driver function to run stage 2: reference selection.

    This stage is processing the metadata file, looks for the output of 
    stages0 and stage1 and checks if a reference file already
    exists.

    It creates a reference frame based on the selection criteria
    defined in the configuration. If no such configuration exists, it 
    falls back to a standard configuration.

    If stage1 has failed to produce output it selects a reference
    based on header information.
    """

    stage2_version = 'stage2 v0.1'

    log = logs.start_stage_log(setup.red_dir, 'stage2', version=stage2_version)
    log.info('Setup:\n' + setup.summary() + '\n')

    # Load all metadata
    reduction_metadata = metadata.load_all_metadata(
        setup.red_dir, 'pyDANDIA_metadata.fits')
    # Check data inventory on metadata

    try:
        n_images = len(reduction_metadata.data_inventory)
    except AttributeError:
        log.info('stage2: data inventory missing.')
        status = 'FAILED'
        report = 'Data inventory (stage1) missing.'
        logs.close_log(log)
        return status, report, reduction_metadata

    # All parameters are part of metadata

    table_structure = [
        ['IMAGE_NAME', 'MOON_STATUS', 'RANKING_KEY'],
        ['S100', 'S100', 'float'],
        ['degree', None,  None]
    ]

    reduction_metadata.create_a_new_layer(layer_name='reference_inventory',
                                          data_structure=table_structure,
                                          data_columns=None)

    log.info('Create reference frame inventory table in metadata')

    # Iterate over images that are in the stage inventory

    reference_ranking = []

    # taking filenames from headers_summary (stage1 change pending)
    filename_images = reduction_metadata.headers_summary[1]['IMAGES']

    for stats_entry in reduction_metadata.imagestats[1]:
        image_filename = stats_entry[0]
        moon_status = 'dark'
        # to be reactivated as soon as it is part of metadata
        # if 'MOONFRAC' in header and 'MOONDIST' in header:
        #    moon_status = moon_brightness_header(header)

        # extract data inventory row for image and calculate sorting key
        ranking_key = add_stage1_rank(reduction_metadata, stats_entry)
        reference_ranking.append([image_filename, ranking_key])
        entry = [image_filename, moon_status, ranking_key]
        reduction_metadata.add_row_to_layer(key_layer='reference_inventory',
                                            new_row=entry)

    # Save the updated layer to the metadata file
    reduction_metadata.save_a_layer_to_file(metadata_directory=setup.red_dir,
                                            metadata_name='pyDANDIA_metadata.fits',
                                            key_layer='reference_inventory')

    # to be completed
    # add ranking for data_inventory images
    for instrument_key in reference_ranking:
        for img_data in reduction_metadata.data_inventory[1]:
            reference_ranking.append(add_stage0_rank(img_data, conf_dict))

    if reference_ranking != []:
        best_image = sorted(reference_ranking, key=itemgetter(1))[-1]
        print best_image
        ref_directory_path = os.path.join(setup.red_dir, 'ref')
        if not os.path.exists(ref_directory_path):
            os.mkdir(ref_directory_path)

        ref_img_path = os.path.join(
            reduction_metadata.data_architecture[1]['IMAGES_PATH'], best_image)

        reduction_metadata.add_column_to_layer('data_architecture',
                                               'REF_PATH', [
                                                   ref_directory_path],
                                               new_column_format=None,
                                               new_column_unit=None)
        reduction_metadata.add_column_to_layer('data_architecture',
                                               'REF_IMAGE', [ref_image_path],
                                               new_column_format=None,
                                               new_column_unit=None)

        status = 'OK'
        report = 'Completed successfully'
        log.info('Updating metadata with info on new images...')
        logs.close_log(log)
        return status, report, reduction_metadata

    else:
        status = 'OK'
        report = 'No suitable image found.'

        log.info('No reference image found...')
        logs.close_log(log)

        return status, report, reduction_metadata


def moon_brightness_header(header):
    '''
    Based on header information, determine if image was
    taken with bright/gray or dark moon
    roughly following the ESO definitions
    https://www.eso.org/sci/observing/phase2/ObsConditions.html
    '''
    if float(header['MOONFRAC']) < 0.4:
        return 'dark'
    else:
        if (float(header['MOONFRAC']) > 0.4 and float(header['MOONFRAC']) < 0.7 and float(header['MOONDIST']) > 90.0):
            return 'gray'
        else:
            return 'bright'


def add_stage1_rank(reduction_metadata, image_stats_entry):
    '''Image ranking based on the data_inventory (stage1 metadata)
    '''
    target_magnitude = 17.  # to be defined in metadata
    magzero_electrons = 1.47371235e+09
    # needs to be updated so that the corresponding values
    # can be configured
    signal_electrons = 10.0**(-0.4 * electrons_per_second_sinistro(
        magzero_electrons, target_magnitude)) * reduction_metadata.reduction_parameters[1]['GAIN'] * reduction_metadata.headers_summary[1]['EXPKEY']
    sky = image_stats_entry['SKY']
    sky_electrons = sky * reduction_metadata.reduction_parameters[1]['GAIN']
    # Simplified number of pixels for optimal aperture photometry
    fwhm_avg = (image_stats_entry['FWHM_X'] **
                2 + image_stats_entry['FWHM_Y']**2)**0.5
    npix = np.pi * \
        reduction_metadata.reduction_paramters[1]['PIXEL_SCALE'] * (
            0.67 * fwhm_avg)**2
    readout_noise = reduction_metadata.reduction_parameters[1]['RON']
    return (signal_electrons) / (signal_electrons + npix * (sky_electrons + readout_noise**2))**0.5


def electrons_per_second_sinistro(mag, magzero_electrons):
    '''
    Temporary definition of electrons per second for a given
    camera. Ideally, the coefficients used here should be
     provided in each instrument file.
    '''
    return magzero_electrons * 10.**(-0.4 * mag)


def check_header_thresholds(moon_status, stage1_entry):
    '''
    Check for images that cannot be used as reference image
    however small the FWHM was. 
    '''
    criteria = [moon_brightness_header != 'bright']
