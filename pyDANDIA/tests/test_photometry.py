# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 14:07:33 2017

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

TEST_DIR = os.path.join(cwd,'data','proc','ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')


def test_run_iterative_PSF_photometry():
    """Function to test the PSF-fitting photometry module for a single image"""
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    log = logs.start_stage_log( cwd, 'test_photometry' )
    
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'reduction_parameters' )

    log.info('Read metadata')
    
    # NOTE: Once stage 2 is complete, the reference image path should be
    # extracted directly from the metadata.
    reduction_metadata.reference_image_path = os.path.join(TEST_DIR,'ref',
                                                           'ref_image.fits')
    ref_image_path = reduction_metadata.reference_image_path
    
    log.info('Performing PSF fitting photometry on '+ref_image_path)

    photometry.run_iterative_PSF_photometry(setup, reduction_metadata,
                                            ref_image_path, log)
    
if __name__ == '__main__':
    
    test_run_iterative_PSF_photometry()
    