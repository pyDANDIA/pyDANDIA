###############################################################################
#
# stage1.py - For a set of images, provide in a metadata fits file:
#                 FWHM (in pix) in x and y directions
#		  sky background
#                 correlation coefficient
#
# Developed by Yiannis Tsapras
# as part of the ROME/REA LCO Key Project.
#
# version 0.1 (development)
#
# Last update: 19 Oct 2017
###############################################################################

from pyDANDIA import  starfind
import os
import sys
from pyDANDIA import  metadata
from pyDANDIA import  logs
from pyDANDIA import  quality_control
from pyDANDIA import  psf

def run_stage1(setup, rerun_all=None):
    """
    Main driver function to run stage 1 of pyDANDIA: measurement of image 
    properties. This stage populates the metadata with the FWHM, sky 
    background, correlation coefficient, number of stars detected and fraction
	of saturated pixels for each image. Updates the 'images_stats' layer in the
    metadata file.

    :param object setup: this is an instance of the ReductionSetup class. See
                         reduction_control.py
    :param boolean rerun_all: Do you want stage1 to rerun on all images? 
                             [True/False]

    :return status, report: two strings reporting whether the stage was 
                            completed successfully
    :rtype string, string
    """

    # Version information

    stage1_version = 'stage1 v0.1'

    # Start logging the steps taken

    log = logs.start_stage_log(setup.red_dir, 'stage1', version=stage1_version)
    log.info('Setup:\n' + setup.summary() + '\n')

    # Attempt to load the metadata file

    reduction_metadata = metadata.MetaData()

    try:
        reduction_metadata.load_all_metadata(metadata_directory=setup.red_dir,
                                             metadata_name='pyDANDIA_metadata.fits')

        logs.ifverbose(
            log, setup, 'Successfully loaded the reduction metadata')
    except:
        log.info('No metadata loaded : check this!')
        status = 'ERROR'
        report = 'Could not load the metadata file.'
        return status, report

    # Find any new images that stage1 needs to process

    all_images = reduction_metadata.find_all_images(setup, reduction_metadata,
                                                    os.path.join(setup.red_dir, 'data'), log=log)

    images = reduction_metadata.find_images_need_to_be_process(setup,
                                                               all_images, stage_number=1,
                                                               rerun_all=rerun_all, log=log)

    full_path_to_images = [os.path.join(
        setup.red_dir, 'data', i) for i in images]

    # If no new images are available for stage1 to process, exit

    if len(images) != 0:
        log.info('Analyzing ' + str(len(images)) + ' images')
    else:
        log.info('No new images for stage1 to process. Exiting stage1.')
        status = 'OK'
        report = 'Completed successfully'
        return status, report

    # Create new layer called 'images_stats' in the metadata file
    # if it doesn't already exist

    if (reduction_metadata.images_stats[1] == None):
        reduction_metadata.create_images_stats_layer()
        log.info('Created images_stats table in metadata')

    log.info('Running starfind on all images')

    # For the set of given images, set the metadata information

    for im in full_path_to_images:
        (status, report, params) = starfind.starfind(setup, im, reduction_metadata,
                                                     plot_it=False, log=log)

        params['fwhm'] = psf.calc_fwhm_from_psf_sigma(params['sigma_x'],
                                                      params['sigma_y'])
        
        # The name of the image

        imname = im.split('/')[-1]
       

        logs.ifverbose(log, setup, 'Processing image %s' % imname)

        (use_phot,use_ref,report) = quality_control.assess_image(reduction_metadata, params, log)
        
        # Add a new row to the images_stats layer
        # (if it doesn't already exist)

        entry = [ imname, params['sigma_x'], params['sigma_y'], params['fwhm'], params['sky'], params['corr_xy'], params['nstars'],
                  params['sat_frac'], params['symmetry'], use_phot, use_ref, ]

        # filling missing values
        for missing in range(len(reduction_metadata.images_stats[1].columns[0:]) - len(entry)):

             entry.append(0)

        reduction_metadata.add_row_to_layer(key_layer='images_stats',
                                            new_row=entry)

    log.info('Updated the images_stats table in metadata')

    # Update the REDUCTION_STATUS table in metadata for stage 1

    reduction_metadata.update_reduction_metadata_reduction_status(images,
                                                                  stage_number=1, status=1, log=log)

    # Save updated metadata

    reduction_metadata.save_updated_metadata(metadata_directory=setup.red_dir,
                                             metadata_name='pyDANDIA_metadata.fits',
                                             log=log)

    (status,report) = quality_control.verify_stage1_output(setup,log)
    
    
    logs.close_log(log)
    
    return status, report
