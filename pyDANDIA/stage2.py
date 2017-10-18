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

    # Create or load the metadata file
    reduction_metadata = stage0.create_or_load_the_reduction_metadata(
        setup.red_dir,
        metadata_name='pyDANDIA_metadata.fits',
        verbose=True, log=log)

    # Check data inventory on metadata

    try:
        n_images = len(reduction_metadata.data_inventory)
    except AttributeError:
        log.info('stage2: data inventory missing.')
        status = 'FAILED'
        report = 'Data inventory (stage1) missing.'
        return status, report, reduction_metadata

    # The configuration file should specify basic parameters for the selection
    config_file_path = os.path.join(setup.pipeline_config_dir, 'config.json')
    conf_dict = config_utils.read_config(config_file_path)
    print conf_dict['gain']['value']
    print conf_dict['ron']['value']

    gain = conf_dict['gain']['value']
    read_noise = conf_dict['ron']['value']

    table_structure = [
        ['INSTRUMENT', 'FILTER', 'MOONDIST',
            'MOONFRAC', 'MOONSTATUS', 'RANKING_KEY'],
        ['S100', 'S100', 'float', 'float', 'S100'],
        ['degree', None,  None]
    ]

    reduction_metadata.create_a_new_layer(layer_name='reference_inventory',
                                          data_structure=table_structure,
                                          data_columns=None)

    log.info('Create reference frame inventory table in metadata')

    #  Iterate over images, check that theyhave been processed by stage1
    #  open image header, open corresponding config
    #  prepare reference for each instrument, filter and binning

    filename_images = glob.glob(os.path.join(setup.red_dir, 'data', '*fits'))
    reference_ranking = {}

    # dictionary for accessing inventory entries with a filename as key
    filename_dict = {key: i for i, key in enumerate(
        reduction_metadata.data_inventory[1]['IM_NAME'])}

    instrument_config_dictionary = {}

    for image_filename in filename_images:

        # The name of the image
        rootname = image_filename.split('/')[-1]
        if rootname in reduction_metadata.data_inventory[1]['IM_NAME']:

            hdulist = fits.open(
                path.join(image_filename, setup.red_dir, 'data', '*fits'))
            header = hdulist[0].header
            hdulist.closed()

            if 'MOONFRAC' in header and 'MOONDIST' in header:
                moon = moon_brightness_header(header)
            else:
                moon = 'dark'

            # extract data inventory row for image and calculate sorting key
            inventory_entry = reduction_metadata.data_inventory[1][filename_dict[image_filename]]
            instrument_file_path = os.path.join(
                setup.pipeline_config_dir, 'config_fl03.json')
            instrument_dict = config_utils.read_config(config_file_path)

            ranking_key = add_stage1_rank(inventory_entry, conf_dict)

            instrument_filter_key = header['INSTRUME'] + \
                '_' + header['FILTER2']
            if not instrument_filter_key in reference_ranking:
                reference_ranking[instrument_filter_key] = [ranking_key]
            else:
                reference_ranking[instrument_filter_key].append(ranking_key)

            if not header['INSTRUME'] in instrument_config_dictionary:
                instrument_file_path = os.path.join(
                    setup.pipeline_config_dir, 'config_' + header['INSTRUME'] + '.json')
                instrument_dict = config_utils.read_config(config_file_path)
            # Add a new row to the reference inventory layer
            # instrument and filter keyword should come from config, tbfixed...

                entry = [image_filename,
                         header['INSTRUME'],
                         header['FILTER2'],
                         header['MOONDIST'],
                         header['MOONFRAC'],
                         header['MOONSTATUS'],
                         ranking_key
                         ]

        print(entry)
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

        # reduction_metadata.save_updated_metadata(
        # reduction_metadata.data_architecture[1]['REF_DIRECTORY_'][0],
        # log=log)
        status = 'OK'
        report = 'Completed successfully'
        log.info('Updating metadata with info on new images...')
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


def add_header_rank(image_name, header, data_inventory):
    '''Image ranking based on header values
    '''
    magzero_electrons = 1.47371235e+09
    if check_header_thresholds(header) == True:
        signal_electrons = 10.0**(-0.4 * electrons_per_second_sinistro(
            target_magnitude)) * header['EXPTIME'] * header['GAIN']
        sky = header['L1MEDIAN']
        sky_electrons = sky * header['GAIN']
        # Simplified number of pixels for optimal aperture photometry
        npix = np.pi * header['PIXSCALE'] * (0.67 * header['L1FWHM'])**2
        readout_noise = header['RDNOISE']
        signal_to_noise_metric.append([image_name, (signal_electrons) / (
            signal_electrons + npix * (sky_electrons + readout_noise**2))**0.5])


def add_stage1_rank(data_inventory_entry, instrument_dict):
    '''Image ranking based on the data_inventory (stage1 metadata)
    '''
    target_magnitude = 17.  # TO BE PART OF CONF DICT
    magzero_electrons = 1.47371235e+09
    # needs to be updated so that the corresponding values
    # can be configured
    signal_electrons = 10.0**(-0.4 * electrons_per_second_sinistro(
        magzero_electrons, target_magnitude)) * instrument_dict['gain']['value']  # * exptime tbd
    sky = data_inventory_entry['SKY']
    sky_electrons = sky * instrument_dict['gain']['value']
    # Simplified number of pixels for optimal aperture photometry
    fwhm_avg = (data_inventory_entry['FWHM_X'] **
                2 + data_inventory_entry['FWHM_Y']**2)**0.5
    npix = np.pi * \
        instrument_dict['pixeL_scale']['value'] * (0.67 * fwhm_avg)**2
    readout_noise = instrument_dict['ron']['value']
    signal_to_noise_metric.append([data_inventory_entry['IM_NAME'], (signal_electrons) / (
        signal_electrons + npix * (sky_electrons + readout_noise**2))**0.5])


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
    criteria = [moon_brightness_header != 'bright', int(
        header['NPFWHM']) > 25, float(header['ELLIP']) > 0.9]
