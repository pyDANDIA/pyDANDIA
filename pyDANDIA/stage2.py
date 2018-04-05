# -*- coding: utf-8 -*-
"""

"""
######################################################################
#
# stage2.py - Second stage of the pipeline.
#             Reads metadata and selects a reference image
#
# dependencies:
#      numpy 1.8+
#      astropy 1.0+
######################################################################

from operator import itemgetter
import os
import sys
import numpy as np

from astropy.io import fits
from astropy.table import Table
from shutil import copyfile

import config_utils

import metadata
import pixelmasks
import logs


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

    It always re-runs when called, since it is a lightweight function
    """

    stage2_version = 'stage2 v0.1'

    log = logs.start_stage_log(setup.red_dir, 'stage2', version=stage2_version)
    log.info('Setup:\n' + setup.summary() + '\n')

    reduction_metadata = metadata.MetaData()

    # Load all metadata
    try:
        reduction_metadata.load_all_metadata(metadata_directory=setup.red_dir,
                                             metadata_name='pyDANDIA_metadata.fits')

    # Check data inventory on metadata
        log.info('stage2 has loaded the reduction metadata')
    except Exception as estr:
        log.info('Could not load metadata!' + repr(estr))
        status = 'FAILED'
        report = 'Loading metadata failed:' + repr(estr)
        return status, report

    try:
        n_images = len(reduction_metadata.images_stats)
    except AttributeError:
        log.info('stage2: data inventory missing.')
        status = 'FAILED'
        report = 'Data inventory (stage1) missing.'
        logs.close_log(log)
        return status, report

    # All parameters are part of metadata

    table_structure = [
        ['IMAGE_NAME', 'MOON_STATUS', 'RANKING_KEY'],
        ['S100', 'S100', 'float'],
        ['degree', None, None]
    ]

    all_images = reduction_metadata.find_all_images(setup, reduction_metadata, os.path.join(setup.red_dir, 'data'), log=log)

    reduction_metadata.create_a_new_layer(layer_name='reference_inventory',
                                          data_structure=table_structure,
                                          data_columns=None)

    log.info('Create reference frame inventory table in metadata')

    # Iterate over images that are in the stage inventory

    reference_ranking = []

    # taking filenames from headers_summary (stage1 change pending)
    filename_images = reduction_metadata.images_stats[1]['IM_NAME']

    for stats_entry in reduction_metadata.images_stats[1]:
        image_filename = stats_entry[0]
        moon_status = 'dark'
        # to be reactivated as soon as it is part of metadata
        # if 'MOONFRAC' in header and 'MOONDIST' in header:
        #    moon_status = moon_brightness_header(header)
        fwhm_arcsec = (float(stats_entry['FWHM_X']) ** 2 + float(stats_entry['FWHM_Y'])**2)**0.5 * float(reduction_metadata.reduction_parameters[1]['PIX_SCALE']) 
        # extract data inventory row for image and calculate sorting key
        # if a sufficient number of stars has been detected at s1 (40)
        if int(stats_entry['NSTARS'])>34 and fwhm_arcsec<3.:
            ranking_key = add_stage1_rank(reduction_metadata, stats_entry)
            reference_ranking.append([image_filename, ranking_key])
            entry = [image_filename, moon_status, ranking_key]
            reduction_metadata.add_row_to_layer(key_layer='reference_inventory',
                                                new_row=entry)

    # Save the updated layer to the metadata file
    reduction_metadata.save_a_layer_to_file(metadata_directory=setup.red_dir,
                                            metadata_name='pyDANDIA_metadata.fits',
                                            key_layer='reference_inventory')

    if reference_ranking != []:
        best_image = sorted(reference_ranking, key=itemgetter(1))[-1]
        ref_directory_path = os.path.join(setup.red_dir, 'ref')
        if not os.path.exists(ref_directory_path):
            os.mkdir(ref_directory_path)

        ref_img_path = os.path.join(
            str(reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]), best_image[0])
        print 'New reference ', best_image[0], ' in ', ref_img_path

        
        copyfile(reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]+'/'+best_image[0],ref_directory_path+'/'+best_image[0])
        #try:
        #    os.symlink(reduction_metadata.data_architecture[1]['IMAGES_PATH'][0]+'/'+best_image[0],ref_directory_path+'/'+best_image[0])
        #except:
        #    print('soft link failed: ',best_image[0])
        if not 'REF_PATH' in reduction_metadata.data_architecture[1].keys():
            reduction_metadata.add_column_to_layer('data_architecture',
                                                   'REF_PATH', [ref_directory_path],
                                                    new_column_format=None,
                                                    new_column_unit=None)
        else:
            reduction_metadata.update_a_cell_to_layer('data_architecture', 0,'REF_PATH', ref_directory_path)
        if  not 'REF_IMAGE' in reduction_metadata.data_architecture[1].keys():
            reduction_metadata.add_column_to_layer('data_architecture',
                                                   'REF_IMAGE', [os.path.basename(ref_img_path)],
                                                    new_column_format=None,
                                                   new_column_unit=None)
        else:
            reduction_metadata.update_a_cell_to_layer('data_architecture', 0,'REF_IMAGE', os.path.basename(ref_img_path))
        # Update the REDUCTION_STATUS table in metadata for stage 2
       
        reduction_metadata.update_reduction_metadata_reduction_status(all_images,
                                                          stage_number=1, status=1, log=log)
        reduction_metadata.save_updated_metadata(metadata_directory=setup.red_dir,
                                                 metadata_name='pyDANDIA_metadata.fits')


        status = 'OK'
        report = 'Completed successfully'
        log.info('Updating metadata with info on new images...')
        logs.close_log(log)

        return status, report

    else:
        status = 'FAILED'
        report = 'No suitable image found.'

        log.info('No reference image found...')
        logs.close_log(log)

        return status, report


def moon_brightness_header(header):
    """
    Based on header information, determine if image was
    taken with bright/gray or dark moon
    roughly following the ESO definitions
    https://www.eso.org/sci/observing/phase2/ObsConditions.html

    :param list header: header or metadata dictionary
    """

    if float(header['MOONFRAC']) < 0.4:
        return 'dark'
    else:
        if float(header['MOONFRAC']) > 0.4 and float(header['MOONFRAC']) < 0.7 and float(header['MOONDIST']) > 90.0:
            return 'gray'
        else:
            return 'bright'


def add_stage1_rank(reduction_metadata, image_stats_entry):
    """
    Image ranking based on the data_inventory (stage1 metadata)

    :param object reduction_metadata: the metadata object

    """
    target_magnitude = 19.  # to be defined in metadata, optimally suited for
                            # fainter mag 19 stars (sky more relevant)
    magzero_electrons = 1.47371235e+09

    # finding the index of the stats_entry image in the headers
    header_idx = list(reduction_metadata.headers_summary[1]['IMAGES']).index(
        image_stats_entry['IM_NAME'])
    # needs to be updated so that the corresponding values
    # can be configured
    signal_electrons = 10.0**(-0.4 * electrons_per_second_sinistro(
        magzero_electrons, target_magnitude)) * float(reduction_metadata.reduction_parameters[1]['GAIN']) * float(reduction_metadata.headers_summary[1][header_idx]['EXPKEY'])
    sky = float(image_stats_entry['SKY'])
    sky_electrons = sky * \
        float(reduction_metadata.reduction_parameters[1]['GAIN'])
    # Simplified number of pixels for optimal aperture photometry
    fwhm_avg = (float(image_stats_entry['FWHM_X']) **
                2 + float(image_stats_entry['FWHM_Y'])**2)**0.5
    npix = np.pi * \
        float(reduction_metadata.reduction_parameters[1]['PIX_SCALE']) * (
            0.67 * fwhm_avg)**2

    readout_noise = float(reduction_metadata.reduction_parameters[1]['RON'])
    return (signal_electrons) / (signal_electrons + npix * (sky_electrons + readout_noise**2))**0.5

def add_stage1_rank_sharpest(reduction_metadata, image_stats_entry):
    """
    Image ranking based on the data_inventory (stage1 metadata)

    :param object reduction_metadata: the metadata object

    """
    # finding the index of the stats_entry image in the headers
    header_idx = list(reduction_metadata.headers_summary[1]['IMAGES']).index(
        image_stats_entry['IM_NAME'])
    fwhm_avg = (float(image_stats_entry['FWHM_X']) **
                2 + float(image_stats_entry['FWHM_Y'])**2)**0.5
    return 1./fwhm_avg

def electrons_per_second_sinistro(mag, magzero_electrons):
    """
    Temporary definition of electrons per second for a given
    camera. Ideally, the coefficients used here should be
    provided in each instrument file.
    :param object mag: the metadata object
    """
    return magzero_electrons * 10.**(-0.4 * mag)


def check_header_thresholds(moon_status, stage1_entry):
    """
    Check for images that cannot be used as reference image
    however small the FWHM was. 
    """
    criteria = [moon_brightness_header != 'bright']
