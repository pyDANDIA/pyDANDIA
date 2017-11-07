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

def test_extract_sub_stamp():
    """Function to test the extraction of substamps from an existing stamp"""

    stamp_dims = (20,20)
    
    image_file = os.path.join(TEST_DATA, 
                            'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
                              
    image = fits.getdata(image_file)
    
    stamp_centres = np.array([[250,250]])
    
    stamps = psf.cut_image_stamps(image, stamp_centres, stamp_dims)
        
    substamp_centres = [ [5,5], [18,18], [5,18], [18,5], [10,10] ]
    
    dx = 10
    dy = 10
    
    for i,location in enumerate(substamp_centres):
        
        xcen = location[0]
        ycen = location[1]
        
        (substamp,corners) = psf.extract_sub_stamp(stamps[0],xcen,ycen,dx,dy)

        assert type(substamp) == type(stamps[0])
        
        hdu = fits.PrimaryHDU(substamp.data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(os.path.join(TEST_DATA,'substamp'+str(i)+'.fits'),
                                     overwrite=True)
        
def test_fit_star_existing_model():
    """Function to test the function of fitting a pre-existing PSF and sky-
    background model to an image at a stars known location, optimizing just for
    the star's intensity rather than for all parameters."""
    
    image_file = os.path.join(TEST_DATA, 
                            'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
                              
    image = fits.getdata(image_file)
    
    psf_model = psf.Moffat2D()
    x_cen = 195.0
    y_cen = 181.0
    psf_radius = 8.0
    psf_params = [ 103301.241291, x_cen, y_cen, 226.750731765,
                  13004.8930993, 103323.763627 ]
    psf_model.update_psf_parameters(psf_params)
    
    sky_model = psf.ConstantBackground()
    sky_model.constant = 1345.0
    
    fitted_model = psf.fit_star_existing_model(image, x_cen, y_cen, psf_radius, 
                                psf_model, sky_model)
    fitted_params = fitted_model.get_parameters()
       
    for i in range(3,5,1):
        
        assert fitted_params[i] == psf_params[i]

def test_subtract_companions_from_psf_stamps():
    """Function to test the function which removes companion stars from the 
    surrounds of a PSF star in a PSF star stamp image."""

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    log = logs.start_stage_log( cwd, 'test_subtract_companions' )
    
    log.info(setup.summary())
    
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'reduction_parameters' )

    star_catalog_file = os.path.join(TEST_DATA,'star_catalog.fits')
                            
    ref_star_catalog = catalog_utils.read_ref_star_catalog_file(star_catalog_file)
    
    log.info('Read in catalog of '+str(len(ref_star_catalog))+' stars')
    
    image_file = os.path.join(TEST_DATA, 
                            'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
                              
    image = fits.getdata(image_file)
    
    stamp_centres = np.array([[258,122]])
    
    stamp_dims = (20,20)
        
    stamps = psf.cut_image_stamps(image, stamp_centres, stamp_dims)

    psf_model = psf.Moffat2D()
    x_cen = 195.0
    y_cen = 181.0
    psf_radius = 8.0
    psf_params = [ 103301.241291, x_cen, y_cen, 226.750731765,
                  13004.8930993, 103323.763627 ]
    psf_model.update_psf_parameters(psf_params)
    
    sky_model = psf.ConstantBackground()
    sky_model.constant = 1345.0
    
    clean_stamps = subtract_companions_from_psf_stamps(setup, reduction_metadata, log, 
                                        ref_star_catalog, stamps,
                                        psf_model,sky_model)
    
    for i, s in enumerate(clean_stamps):
        
        hdu = fits.PrimaryHDU(s.data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(os.path.join(TEST_DATA,'clean_stamp'+str(i)+'.fits'),
                                     overwrite=True)

    logs.close_log(log)


if __name__ == '__main__':
    
    test_cut_image_stamps()
    #test_build_psf()
    test_extract_sub_stamp()
    test_fit_star_existing_model()