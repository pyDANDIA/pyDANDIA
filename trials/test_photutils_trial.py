# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:35:52 2017

@author: rstreet
"""


import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import numpy as np
import logs
import pipeline_setup
import metadata
import photometry
from astropy.table import Table

TEST_DIR = os.path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')


def test_run_iterative_PSF_photometry():
    """Function to test the PSF-fitting photometry module for a single image"""
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    log = logs.start_stage_log( cwd, 'test_photometry' )
    
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'reduction_parameters' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'images_stats' )

    log.info('Read metadata')
    
    # NOTE: Once stage 2 is complete, the reference image path should be
    # extracted directly from the metadata.
    reduction_metadata.reference_image_path = os.path.join(TEST_DIR,'data',
                                                           'lsc1m005-fl15-20170418-0131-e91_cropped.fits')
    image_path = reduction_metadata.reference_image_path
    
    log.info('Performing PSF fitting photometry on '+os.path.basename(image_path))

    phot_data = photometry.run_iterative_PSF_photometry(setup, reduction_metadata,
                                            image_path, log, diagnostics=True)
    
    test_output = Table()
    
    assert type(phot_data) == type(test_output)
    
    logs.close_log(log)
    
if __name__ == '__main__':
    
    test_run_iterative_PSF_photometry()
    