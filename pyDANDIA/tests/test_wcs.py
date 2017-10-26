# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:14:17 2017

@author: rstreet
"""
from os import getcwd, path, remove
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import logs
import wcs
import stage3
from astropy.io import fits
import catalog_utils

cwd = getcwd()
TEST_DATA = path.join(cwd,'data')

def test_reference_astrometry():
    
    log = logs.start_stage_log( cwd, 'test_wcs' )
    
    image_path = path.join(TEST_DATA,'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
    detected_sources_file = path.join(TEST_DATA,'lsc1m005-fl15-20170701-0144-e91_cropped_sources.txt')
    
    outputs = ['reference_detected_sources_pixels.png',
               'reference_detected_sources_world.png',
               'astrometry_separations.png',
               'star_catalog.fits']    
    
    for item in outputs:
        if path.isfile(path.join(TEST_DATA,item)):
            remove(path.join(TEST_DATA,item))
    
    detected_sources = catalog_utils.read_source_catalog(detected_sources_file)
    
    ref_source_catalog = wcs.reference_astrometry(log,image_path,detected_sources)
    
    assert path.isfile(path.join(TEST_DATA,'reference_detected_sources_pixels.png')) == True
    assert path.isfile(path.join(TEST_DATA,'reference_detected_sources_world.png')) == True
    assert path.isfile(path.join(TEST_DATA,'astrometry_separations.png')) == True
    assert path.isfile(path.join(TEST_DATA,'star_catalog.fits')) == True
    
    logs.close_log(log)
    

def test_search_vizier_for_objects_in_fov():
    """Function to test the online extraction of a catalogue of known
    sources within a given field of view"""
    
    image = path.join(TEST_DATA,'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
    header = fits.getheader(image)
    radius = ( float(header['pixscale'])*float(header['NAXIS1']) ) / 60.0
    catalog = wcs.search_vizier_for_objects_in_fov(header['RA'], header['Dec'], radius)

    assert len(catalog) == 50
    
if __name__ == '__main__':
    test_reference_astrometry()