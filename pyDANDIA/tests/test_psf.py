# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:26:00 2017

@author: rstreet
"""

import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import matplotlib.pyplot as plt
import numpy as np
import logs
import pipeline_setup
import metadata
import psf
import catalog_utils
from astropy.io import fits
from astropy.nddata import Cutout2D

TEST_DATA = os.path.join(cwd,'data')
TEST_DIR = os.path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')

def test_build_psf():
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    log = logs.start_stage_log( cwd, 'test_build_psf' )
    
    log.info(setup.summary())
    
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'reduction_parameters' )
    
    reduction_metadata.reference_image_path = os.path.join(TEST_DATA,
                                                           'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
    reduction_metadata.background_type = 'constant'
    
    star_catalog_file = os.path.join(TEST_DATA,'star_catalog.fits')
                            
    ref_star_catalog = catalog_utils.read_ref_star_catalog_file(star_catalog_file)
    
    log.info('Read in catalog of '+str(len(ref_star_catalog))+' stars')
    
    psf_stars_idx = np.zeros(len(ref_star_catalog))
    psf_stars_idx[0:100] = 1.0
    
    ref_image = fits.getdata(reduction_metadata.reference_image_path)
    
    log.info('Loaded reference image')
    
    psf.build_psf(setup, reduction_metadata, log, ref_image, ref_star_catalog, 
              psf_stars_idx, diagnostics=True)

    logs.close_log(log)
    
def test_cut_image_stamps():
    
    stamp_dims = (20,20)
    
    image_file = os.path.join(TEST_DATA, 
                            'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
                              
    image = fits.getdata(image_file)
    
    detected_sources_file = os.path.join(TEST_DATA,
                            'lsc1m005-fl15-20170701-0144-e91_cropped_sources.txt')
                            
    detected_sources = catalog_utils.read_source_catalog(detected_sources_file)
        
    stamps = psf.cut_image_stamps(image, detected_sources[0:100,1:3], stamp_dims)
        
    test_stamp = Cutout2D(np.ones([100,100]), (50,50), (10,10))
    
    got_stamp = False
    for s in stamps:
        
        if type(s) == type(test_stamp):
            
            got_stamp = True
    
    assert got_stamp == True

if __name__ == '__main__':
    
    test_cut_image_stamps()
    test_build_psf()
    