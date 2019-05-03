# -*- coding: utf-8 -*-
"""
Created on Wed May  1 16:56:18 2019

@author: rstreet
"""

from os import getcwd, path, remove
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import logs
import wcs
import stage3
import pipeline_setup
import metadata
import match_utils
from astropy.io import fits
from astropy.table import Table, Column
from astropy.wcs import WCS as aWCS
from astropy.coordinates import SkyCoord
from astropy import units
import catalog_utils
import reference_astrometry
import numpy as np 

test_full_frame = True
cwd = getcwd()
TEST_DATA = path.join(cwd,'data')

if test_full_frame:

    TEST_DIR = path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')
else:
    
    TEST_DIR = path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')

def test_run_reference_astrometry():
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    (status, report) = reference_astrometry.run_reference_astrometry(setup)
    
def test_detect_objects_in_reference_image():
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})    
    
    log = logs.start_stage_log( cwd, 'test_wcs' )
    
    meta = metadata.MetaData()
    meta.load_a_layer_from_file( setup.red_dir, 
                                'pyDANDIA_metadata.fits', 
                                'data_architecture' )
    meta.load_a_layer_from_file( setup.red_dir, 
                                'pyDANDIA_metadata.fits', 
                                'images_stats' )
    meta.load_a_layer_from_file( setup.red_dir, 
                                'pyDANDIA_metadata.fits', 
                                'reduction_parameters' )
    meta.load_a_layer_from_file( setup.red_dir, 
                                'pyDANDIA_metadata.fits', 
                                'headers_summary' )
    
    det_catalog_file = path.join(setup.red_dir,'ref', 'detected_stars_full.reg')
    if path.isfile(det_catalog_file):
        remove(det_catalog_file)
    
    meta_pars = stage3.extract_parameters_stage3(meta, log)
    
    image_path = path.join(TEST_DIR, 'ref', 'lsc1m005-fl15-20170418-0131-e91_cropped.fits')
    
    header = fits.getheader(image_path)
    
    image_wcs = aWCS(header)
    
    detected_sources = reference_astrometry.detect_objects_in_reference_image(setup, 
                                                                              meta,
                                                                              meta_pars,
                                                                              image_wcs,
                                                                              log)
    
    assert type(detected_sources) == type(Table())
    assert path.isfile(det_catalog_file)
    
    logs.close_log(log)

def test_catalog_objects_in_reference_image():
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})    
    
    log = logs.start_stage_log( cwd, 'test_wcs' )
    
    cat_catalog_file = path.join(setup.red_dir,'ref', 'catalog_stars_full.reg')
    if path.isfile(cat_catalog_file):
        remove(cat_catalog_file)
    
    image_path = path.join(TEST_DIR, 'ref', 'lsc1m005-fl15-20170418-0131-e91_cropped.fits')
    
    header = fits.getheader(image_path)
    
    image_wcs = aWCS(header)
    
    gaia_sources = reference_astrometry.catalog_objects_in_reference_image(setup, 
                                                                           header, 
                                                                           image_wcs, 
                                                                           log)
    
    assert type(gaia_sources) == type(Table())
    assert path.isfile(cat_catalog_file)
    
    logs.close_log(log)

def test_phot_catalog_objects_in_reference_image():
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})    
    
    log = logs.start_stage_log( cwd, 'test_wcs' )
    
    image_path = path.join(TEST_DIR, 'ref', 'lsc1m005-fl15-20170418-0131-e91_cropped.fits')
    
    header = fits.getheader(image_path)
    
    image_wcs = aWCS(header)

    source_cat = reference_astrometry.phot_catalog_objects_in_reference_image(setup, header, image_wcs, log)
    
    print(source_cat)
    
    assert type(source_cat) == type(Table())
    assert len(source_cat) > 0
    
    logs.close_log(log)
    
if __name__ == '__main__':
    
    #test_detect_objects_in_reference_image()
    #test_catalog_objects_in_reference_image()
    #test_phot_catalog_objects_in_reference_image()
    test_run_reference_astrometry()
    