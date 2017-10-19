# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 15:47:08 2017

@author: rstreet
"""

from os import getcwd, path, remove
from sys import exit
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import numpy as np
import logs
import metadata
import pipeline_setup
import catalog_utils
import sky_background

cwd = getcwd()
TEST_DATA = path.join(cwd,'data')
TEST_DIR = path.join(cwd,'data','proc','ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')

def test_model_sky_background():
    """Function to test the function to model the sky background of an image"""
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    log = logs.start_stage_log( cwd, 'test_psf_selection' )
    
    detected_sources_file = path.join(TEST_DATA,
                                      'lsc1m005-fl15-20170701-0144-e91_cropped_sources.txt')

    detected_sources = catalog_utils.read_source_catalog(detected_sources_file)
    
    ref_star_catalog = np.zeros([len(detected_sources),13])
    ref_star_catalog[:,0] = detected_sources[:,0]
    ref_star_catalog[:,1] = detected_sources[:,1]
    ref_star_catalog[:,2] = detected_sources[:,2]
    
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file(setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'reduction_parameters')
    reduction_metadata.ref_image_path = path.join(TEST_DATA,
                                                  'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
                                                      
    log.info('Read metadata')
    
    
    sky_background.model_sky_background(setup,reduction_metadata,log,ref_star_catalog)
    
    
    logs.close_log(log)

if __name__ == '__main__':
    
    test_model_sky_background()
    