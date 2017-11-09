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
import catalog_utils
import psf
from astropy.table import Table


TEST_DATA = os.path.join(cwd,'data')
TEST_DIR = os.path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')

def test_run_psf_photometry():
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
    reduction_metadata.reference_image_path = os.path.join(TEST_DATA, 
                            'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
                            
    image_path = reduction_metadata.reference_image_path
    
    star_catalog_file = os.path.join(TEST_DATA,'star_catalog.fits')
                            
    ref_star_catalog = catalog_utils.read_ref_star_catalog_file(star_catalog_file)
    
    psf_model = psf.Moffat2D()
    x_cen = 194.654006958
    y_cen = 180.184967041
    psf_radius = 8.0
    psf_params = [ 103301.241291, x_cen, y_cen, 226.750731765,
                  13004.8930993, 103323.763627 ]
    psf_model.update_psf_parameters(psf_params)

    sky_model = psf.ConstantBackground()
    sky_model.background_parameters.constant = 1345.0

    
    
    log.info('Performing PSF fitting photometry on '+os.path.basename(image_path))

    ref_star_catalog = photometry.run_psf_photometry(setup,reduction_metadata,
                                                     log,ref_star_catalog,
                                                     image_path,psf_model,sky_model)
    
    print ref_star_catalog
    
    logs.close_log(log)
    
if __name__ == '__main__':
    
    test_run_psf_photometry()
    