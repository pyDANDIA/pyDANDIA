# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 10:31:30 2019

@author: rstreet
"""

import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import numpy as np
from astropy.table import Table
from astropy.io import fits
import catalog_utils
import wcs
import pipeline_setup

TEST_DIR = os.path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')

def test_output_vizier_catalog():
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
        
    field_name = 'ROME-FIELD-01'
    ra = '17:51:20.6149'
    dec = '-30:03:38.9442'
    radius = 5.0
    
    output_file = os.path.join(setup.pipeline_config_dir,field_name+'_2mass_catalog.fits')
    
    if os.path.isfile(output_file):
        os.remove(output_file)
        
    vizier_result = wcs.search_vizier_for_2mass_sources(ra, dec, radius)
    
    catalog_utils.output_vizier_catalog(output_file, vizier_result, '2MASS')
    
    assert os.path.isfile(output_file)

def test_read_vizier_catalog():
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    cat_file = os.path.join(setup.pipeline_config_dir,'ROME-FIELD-01_2mass_catalog.fits')
    
    cat_table = catalog_utils.read_vizier_catalog(cat_file,'2MASS')
    
    assert type(cat_table) == type(Table())
    
if __name__ == '__main__':
    
    #test_output_vizier_catalog()
    test_read_vizier_catalog()