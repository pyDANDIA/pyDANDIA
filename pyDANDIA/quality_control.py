# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 13:34:31 2018

@author: rstreet
"""
import os
from astropy.io import fits
import metadata

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
    
def assess_image(reduction_metadata,image_params,log):
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
    
    Flag convention: 1 = OK, 0 = bad image
    """
    
    use_phot = 1
    use_ref = 1
    report = ''
    
    fwhm_max = reduction_metadata.reduction_parameters[1]['MAX_FWHM_ARCSEC'][0]
    sky_max = reduction_metadata.reduction_parameters[1]['MAX_SKY'][0]
    sky_ref_max = reduction_metadata.reduction_parameters[1]['MAX_SKY_REF'][0]
    
    if image_params['nstars'] == 0:
        
        use_phot = 0
        use_ref = 0
        report = append_errors(report, 'No stars detected in frame')
        
    if image_params['fwhm_x'] > fwhm_max or image_params['fwhm_y'] > fwhm_max:
        
        use_phot = 0
        use_ref = 0
        report = append_errors(report, 'FWHM exceeds threshold')
        
    if image_params['sky'] > sky_max:
        
        use_phot = 0
        report = append_errors(report, 'Sky background exceeds threshold for photometry')
    
    if image_params['sky'] > sky_ref_max:
        
        use_ref = 0
        report = append_errors(report, 'Sky background exceeds threshold for reference')
    
    if use_phot == 1 and use_ref == 1 and len(report) == 0:
        report = 'OK'
    
    log.info('Quality assessment:')
    log.info('Use for photometry = '+str(use_phot))
    log.info('Use for reference = '+str(use_ref))
    log.info('Report: '+report)
    
    return use_phot, use_ref, report
    
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
    