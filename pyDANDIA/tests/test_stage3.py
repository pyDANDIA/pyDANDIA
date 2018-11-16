# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:58:40 2017

@author: rstreet
"""

import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import logs
import pipeline_setup
import metadata
import stage3
import starfind
from astropy.io import fits
import numpy as np

TEST_DIR = os.path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')

def test_run_stage3():
    """Function to test the execution of Stage 3 of the pipeline, end-to-end"""
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    (status, report) = stage3.run_stage3(setup)

def test_find_reference_flux():
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    meta = metadata.MetaData()
    meta.load_a_layer_from_file( setup.red_dir, 
                                'pyDANDIA_metadata.fits', 
                                'data_architecture' )
    meta.load_a_layer_from_file( setup.red_dir, 
                                'pyDANDIA_metadata.fits', 
                                'images_stats' )
    
    reference_image_path = os.path.join(str(meta.data_architecture[1]['REF_PATH'][0]),
                                     str(meta.data_architecture[1]['REF_IMAGE'][0]))

    log = logs.start_stage_log( cwd, 'test_stage3' )
    
    scidata = fits.getdata(reference_image_path)
    
    detected_sources = starfind.detect_sources(setup,meta,reference_image_path,scidata,log)

    ref_flux = stage3.find_reference_flux(detected_sources)
    
    assert(type(ref_flux) == type(np.sqrt(9.0)))
    assert(ref_flux > 0.0)
    
if __name__ == '__main__':
    
    #test_run_stage3()
    test_find_reference_flux()
    