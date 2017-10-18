######################################################################
#
# stage1.py - For a set of images, provide in a metadata fits file:
#                 FWHM (in pix) in x and y directions
#		  sky background
#                 correlation coefficient
#
# dependencies:
#       starfind.py
#
# Developed by Yiannis Tsapras
# as part of the ROME/REA LCO Key Project.
#
# version 0.1a (development)
#
# Last update: 11 Oct 2017
###############################################################################

import config_utils
import starfind
import os
import sys
import glob
import metadata
import stage0
import logs


def run_stage1(setup):
    """Main driver function to run stage 1 of pyDANDIA: measurement of image 
    properties. This stage populates the metadata with the FWHM and sky 
    background for each image.
    Input: setup - is an instance of the ReductionSetup class. See 
           reduction_control.py
    Output: updates the 'image_stats' layer in the metadata file
    """

    stage1_version = 'stage1 v0.1'

    log = logs.start_stage_log(setup.red_dir, 'stage1', version=stage1_version)
    log.info('Setup:\n' + setup.summary() + '\n')

    # Load the metadata file

    reduction_metadata = metadata.MetaData()
    try:
        reduction_metadata.load_all_metadata(metadata_directory=setup.red_dir,
                                             metadata_name='pyDANDIA_metadata.fits')
        
        logs.ifverbose(log,setup,'Successfully loaded the reduction metadata')
    except:
        logs.ifverbose(log,setup,'No metadata loaded : check this!')
        
        sys.exit(1)

    # Collect the image files
    #images = glob.glob(os.path.join(setup.red_dir, 'data', '*fits'))

    images = reduction_metadata.find_images_need_to_be_process(setup,
                                     reduction_metadata=reduction_metadata,
                                     list_of_images=setup.red_dir,
                                     log=None)
    
    log.info('Analyzing ' + str(len(images)) + ' images')

    # The configuration file specifies the header information for
    # the input images
    config_file_path = os.path.join(setup.pipeline_config_dir, 'config.json')
    conf_dict = config_utils.read_config(config_file_path)
    gain = conf_dict['gain']['value']
    read_noise = conf_dict['ron']['value']

    # Create new layer called 'image_stats' in the metadata file
    # (if it doesn't already exist)
    table_structure = [
        ['IM_NAME', 'FWHM_X', 'FWHM_Y', 'SKY', 'CORR_XY'],
        ['S100', 'float', 'float', 'float', 'float'],
        [None, 'arcsec', 'arcsec', 'ADU_counts', None]
    ]

    reduction_metadata.create_a_new_layer(layer_name='image_stats',
                                          data_structure=table_structure,
                                          data_columns=None)

    log.info('Created image_stats table in metadata')
    log.info('Running starfind on all images')

    # For the set of given images, set the metadata information
    for im in images:
        (status, report, params) = starfind.starfind(setup, im, plot_it=False,
                                                     log=log)

        # The name of the image
        imname = im.split('/')[-1]

        logs.ifverbose(log,setup,'Processing image %s' % imname)

        # Add a new row to the image_stats layer
        # (if it doesn't already exist)
        entry = [
            imname,
            params['fwhm_x'],
            params['fwhm_y'],
            params['sky'],
            params['corr_xy']
        ]

        reduction_metadata.add_row_to_layer(key_layer='image_stats',
                                            new_row=entry)

    # Save the updated layer to the metadata file
    reduction_metadata.save_a_layer_to_file(metadata_directory=setup.red_dir,
                                            metadata_name='pyDANDIA_metadata.fits',
                                            key_layer='image_stats')

    log.info('Updated the image_stats table in metadata')

    status = 'OK'
    report = 'Completed successfully'
    return status, report
