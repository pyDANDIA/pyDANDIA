# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 13:34:31 2018

@author: rstreet
"""
import os
from astropy.io import fits

def verify_stage0_output(setup,log):
    """Function to verify that stage 0 has produced the expected output. 
    
    This function checks for the presences of a metadata file and a stage log
    """
    
    log.info('Verifying stage 0 data products:')
    
    status = 'OK'
    report = 'Completed successfully'
    
    metadata = os.path.join(setup.red_dir, 'pyDANDIA_metadata.fits')
    stage_log = os.path.join(setup.red_dir, 'stage0.log')
    
    if os.path.isfile(metadata) == False or os.path.isfile(stage_log) == False:
        
        status = 'ERROR'
        report = 'Stage 0 finished without producing its expected data products'
        
        log.info('Status: '+status)
        log.info('Report: '+report)
        
        return status, report
        
    m = fits.open(metadata)
    
    if len(m) != 6:
        
        status = 'ERROR'
        report = 'Stage 0 produced an incomplete metadata file'
        
        log.info('Status: '+status)
        log.info('Report: '+report)
        
        return status, report
    
    log.info('Status: '+status)
    log.info('Report: '+report)
        
    return status, report