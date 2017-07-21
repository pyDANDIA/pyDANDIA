# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 17:14:17 2017

@author: rstreet
"""
from os import getcwd, path
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import logs
import wcs
import stage3
from astropy.io import fits

cwd = getcwd()
TEST_DATA = path.join(cwd,'data')

def test_fit_wcs():
    """Function to test the function for refining an image WCS"""
    
    meta = stage3.TmpMeta(TEST_DATA)
    
    log = logs.start_stage_log( cwd, 'test_wcs' )
    
    input_image_path = path.join(TEST_DATA,'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
    catalog_source = path.join(TEST_DATA,'lsc1m005-fl15-20170701-0144-e91_cropped_2mass.cat')
    detected_sources = path.join(TEST_DATA,'lsc1m005-fl15-20170701-0144-e91_cropped_sources.txt')
    output_image_path = path.join(TEST_DATA,'lsc1m005-fl15-20170701-0144-e91_cropped_wcs.fits')
    
    wcs.run_imwcs(detected_sources,catalog_sources,input_image_path,output_image_path)

    logs.close_stage_log(log)

def test_get_catalog_objects_in_fov():
    """Function to test the online extraction of a catalogue of known
    sources within a given field of view"""
    
    image = path.join(TEST_DATA,'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
    header = fits.getheader(image)
    radius = ( float(header['pixscale'])*float(header['NAXIS1']) ) / 60.0
    catalog = wcs.get_catalog_objects_in_fov(header['RA'], header['Dec'], radius)

    assert len(catalog) == 50
    
if __name__ == '__main__':
    test_get_catalog_objects_in_fov()