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
import psf_selection
import random
import pipeline_setup
import metadata

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

    ref_image_path = reduction_metadata.reduction_parameters[1]['REF_IMAGE_PATH'][0]
    
    log.info('Performing PSF fitting photometry on '+ref_image_path)

    photometry.run_iterative_PSF_photometry(setup, reduction_metadata,
                                            ref_image_path, log)
    
if __name__ == '__main__':
    
    test_run_iterative_PSF_photometry()
    