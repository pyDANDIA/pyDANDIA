# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:07:39 2017

@author: rstreet
"""

import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import pipeline_setup
import metadata
import stage3
import logs

TEST_DIR = os.path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')
                        
def test_extract_parameters_stage3():
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    log = logs.start_stage_log( cwd, 'test_photometry' )
    
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'reduction_parameters' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'images_stats' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'data_architecture' )
    reduction_metadata.reference_image_path = os.path.join(setup.red_dir,
                                              'lsc1m005-fl15-20170418-0131-e91_cropped.fits')
                                            
    meta_pars = stage3.extract_parameters_stage3(reduction_metadata,log)
    
    print meta_pars

    logs.close_log(log)
    
def test_get_gain():
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    reduction_metadata = metadata.MetaData()
    
    gain = reduction_metadata.get_gain()
    
    assert gain == None
    
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'reduction_parameters' )
    
    gain = reduction_metadata.get_gain()
    
    assert type(gain) == type(1.0)

def test_extract_exptime():
    
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'data_architecture' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'headers_summary' )
    
    image_name = reduction_metadata.data_architecture[1]['REF_IMAGE'][0]

    exp_time = reduction_metadata.extract_exptime(image_name)
    
    assert(exp_time == 300.0)
    
if __name__ == '__main__':
    
    test_extract_parameters_stage3()
    test_get_gain()
    test_extract_exptime()
    