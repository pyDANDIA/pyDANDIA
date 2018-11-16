# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 11:43:07 2017

@author: rstreet
"""

from os import getcwd, path
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import starfind
import metadata
import stage3
import logs
from astropy.io import fits
import pipeline_setup
import numpy as np

cwd = getcwd()
TEST_DATA = path.join(cwd,'data')

TEST_DIR = path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')
                        
def test_star_finder(output=False):
    """Function to test the performance of the starfind function"""
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    meta = metadata.MetaData()
    meta.load_a_layer_from_file( setup.red_dir, 
                                'pyDANDIA_metadata.fits', 
                                'data_architecture' )
    meta.load_a_layer_from_file( setup.red_dir, 
                                'pyDANDIA_metadata.fits', 
                                'images_stats' )
    
    reference_image_path = path.join(str(meta.data_architecture[1]['REF_PATH'][0]),
                                     str(meta.data_architecture[1]['REF_IMAGE'][0]))

    log = logs.start_stage_log( cwd, 'test_starfind' )
    
    log.info(setup.summary())
    
    scidata = fits.getdata(reference_image_path)
    
    sources = starfind.detect_sources(setup,meta,reference_image_path,scidata,log)

    assert len(sources) == 986
    assert type(sources) == type(np.zeros([10]))
    
    logs.close_stage_log(log)
    
if __name__ == '__main__':
    test_star_finder(output=True)