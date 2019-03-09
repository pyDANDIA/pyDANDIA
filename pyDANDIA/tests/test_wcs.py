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
import pipeline_setup
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS as aWCS
import catalog_utils

cwd = getcwd()
TEST_DATA = path.join(cwd,'data')

TEST_DIR = path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')
                        
def test_reference_astrometry():

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})    
    
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
    
    ref_source_catalog = wcs.reference_astrometry(setup,log,image_path,detected_sources)
    
    assert path.isfile(path.join(TEST_DATA,'ref','reference_detected_sources_pixels.png')) == True
    assert path.isfile(path.join(TEST_DATA,'ref','reference_detected_sources_world.png')) == True
    assert path.isfile(path.join(TEST_DATA,'ref','astrometry_separations.png')) == True
    assert path.isfile(path.join(TEST_DATA,'ref','star_catalog.fits')) == True
    assert path.isfile(path.join(TEST_DATA,'ref','ref_image_wcs.fits')) == True
    
    logs.close_log(log)
    

def test_search_vizier_for_objects_in_fov():
    """Function to test the online extraction of a catalogue of known
    sources within a given field of view"""
    
    image = path.join(TEST_DATA,'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
    header = fits.getheader(image)
    radius = ( float(header['pixscale'])*float(header['NAXIS1']) ) / 60.0
    catalog = wcs.search_vizier_for_objects_in_fov(header['RA'], header['Dec'], radius)

    assert len(catalog) == 50

def test_search_vizier_for_2mass_sources():
    """Function to test the online extraction of a catalogue of known
    sources within a given field of view"""
    
    ra = '270.8210141'
    dec = '-26.909889'
    radius = 2027*0.467*60.0/2.0
    catalog = wcs.search_vizier_for_2mass_sources(ra, dec, radius)

    assert len(catalog) > 0

def test_search_vizier_for_gaia_sources():
    """Function to test the online extraction of a catalogue of known
    sources within a given field of view"""
    
    ra = '270.8210141'
    dec = '-26.909889'
    radius = 2027*0.467*60.0/2.0
    catalog = wcs.search_vizier_for_gaia_sources(ra, dec, radius)

    assert len(catalog) > 0
    
def test_fetch_catalog_sources_for_field():
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})    
    
    log = logs.start_stage_log( cwd, 'test_wcs' )
    
    field = 'ROME-FIELD-01'
    header = {'CRPIX1': 199.8948336988849,
                'CRPIX2': 199.7880003717081,
                'CRVAL1': 269.5375,             
                'CRVAL2': -28.01805555555556,          
                'CUNIT1': 'deg     ',                             
                'CUNIT2': 'deg     ',                                 
                'CD1_1': 0.0001081,                 
                'CD1_2': 0.0000000,              
                'CD2_1': 0.0000000,           
                'CD2_2': -0.0001081,
                'NAXIS1': 4096,
                'PIXSCALE': 0.26}
                
    image_wcs = aWCS(header)
    
    catalog_sources = wcs.fetch_catalog_sources_for_field(setup,field,header,image_wcs,log)
    
    assert type(catalog_sources) == type(Table())
    assert len(catalog_sources) > 0
    
    field = 'ROME-FIELD-21'
    catalog_sources = wcs.fetch_catalog_sources_for_field(setup,field,header,image_wcs,log)
    
    assert type(catalog_sources) == type(Table())
    assert len(catalog_sources) > 0
    
    logs.close_log(log)
    
if __name__ == '__main__':

    test_reference_astrometry()
    #test_search_vizier_for_2mass_sources()
    #test_fetch_catalog_sources_for_field()
    #test_search_vizier_for_gaia_sources()
    