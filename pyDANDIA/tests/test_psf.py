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
from astropy import visualization
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
    psf_stars_idx[400:500] = 1
    ref_star_catalog[:,13] = psf_stars_idx

    ref_image = fits.getdata(reduction_metadata.reference_image_path)
    
    log.info('Loaded reference image')
    
    sky_model = psf.ConstantBackground()
    sky_model.constant = 1345.0
    sky_model.background_parameters.constant = 1345.0

    (psf_model, status) = psf.build_psf(setup, reduction_metadata, log, ref_image, 
                              ref_star_catalog, sky_model, diagnostics=True)
    
    logs.close_log(log)
    
def test_cut_image_stamps():

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DATA})
    
    stamp_dims = (20,20)
    
    image_file = os.path.join(TEST_DATA, 
                            'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
                              
    image = fits.getdata(image_file)
    
    detected_sources_file = os.path.join(TEST_DATA,
                            'lsc1m005-fl15-20170701-0144-e91_cropped_sources.txt')
                            
    detected_sources = catalog_utils.read_source_catalog(detected_sources_file)
        
    stamps = psf.cut_image_stamps(setup, image, detected_sources[0:100,1:3], stamp_dims)
        
    test_stamp = Cutout2D(np.ones([100,100]), (50,50), (10,10))
    
    got_stamp = False
    for s in stamps:
        
        if type(s) == type(test_stamp):
            
            got_stamp = True
    
    assert got_stamp == True

def test_extract_sub_stamp():
    """Function to test the extraction of substamps from an existing stamp"""

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DATA})
    
    stamp_dims = (20,20)
    
    image_file = os.path.join(TEST_DATA, 
                            'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
                              
    image = fits.getdata(image_file)
    
    stamp_centres = np.array([[250,250]])
    
    stamps = psf.cut_image_stamps(setup, image, stamp_centres, stamp_dims)
        
    substamp_centres = [ [5,5], [18,18], [5,18], [18,5], [10,10] ]
    
    dx = 10
    dy = 10
    
    substamp_dims = []
    for centre in substamp_centres:
        
        xmin = max(0,(centre[0] - int(float(dx)/2.0)))
        xmax = min(stamp_dims[0], (centre[0] + int(float(dx)/2.0)) )
        ymin = max(0,(centre[1] - int(float(dy)/2.0)))
        ymax = min(stamp_dims[1], (centre[1] + int(float(dy)/2.0)) )
        
        substamp_dims.append( (ymax-ymin, xmax-xmin) )
    
    for i,location in enumerate(substamp_centres):
        
        xcen = location[0]
        ycen = location[1]
        
        (substamp,corners) = psf.extract_sub_stamp(stamps[0],xcen,ycen,dx,dy)

        assert type(substamp) == type(stamps[0])
        
        assert substamp.data.shape == substamp_dims[i]
        
        hdu = fits.PrimaryHDU(substamp.data)
        hdulist = fits.HDUList([hdu])
        hdulist.writeto(os.path.join(TEST_DATA,'substamp'+str(i)+'.fits'),
                                     overwrite=True)
        
def test_fit_star_existing_model():
    """Function to test the function of fitting a pre-existing PSF and sky-
    background model to an image at a stars known location, optimizing just for
    the star's intensity rather than for all parameters."""
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DATA})

    image_file = os.path.join(TEST_DATA, 
                            'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
                              
    image = fits.getdata(image_file)
    
    psf_model = psf.Moffat2D()
    x_cen = 194.654006958
    y_cen = 180.184967041
    psf_radius = 8.0
    psf_params = [ 103301.241291, x_cen, y_cen, 226.750731765,
                  13004.8930993, 103323.763627 ]
    psf_model.update_psf_parameters(psf_params)
    
    sky_model = psf.ConstantBackground()
    sky_model.constant = 1345.0
    sky_model.background_parameters.constant = 1345.0
    
    (fitted_model,good_fit) = psf.fit_star_existing_model(setup, image, x_cen, y_cen, 
                                psf_radius, psf_model, sky_model,
                                diagnostics=True)
                                
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
    
    psf_idx = [ 248 ]
    psf_x = 257.656
    psf_y = 121.365
    
    stamp_centres = np.array([[psf_x,psf_y]])
    
    psf_size = 10.0
    stamp_dims = (20,20)
    
    stamps = psf.cut_image_stamps(setup, image, stamp_centres, stamp_dims)
    
    if len(stamps) == 0:
        
        log.info('ERROR: No PSF stamp images returned.  PSF stars too close to the edge?')
    
    else:
        
        for i, s in enumerate(stamps):
            
            fig = plt.figure(1)
                
            norm = visualization.ImageNormalize(s.data, \
                            interval=visualization.ZScaleInterval())
        
            plt.imshow(s.data, origin='lower', cmap=plt.cm.viridis, 
                           norm=norm)
                
            plt.xlabel('X pixel')
        
            plt.ylabel('Y pixel')
            
            plt.axis('equal')
            
            plt.savefig(os.path.join(setup.red_dir,'psf_star_stamp'+str(i)+'.png'))
    
            plt.close(1)
            
        psf_model = psf.Moffat2D()
        x_cen = psf_size + (psf_x-int(psf_x))
        y_cen = psf_size + (psf_x-int(psf_y))
        psf_radius = 8.0
        psf_params = [ 103301.241291, x_cen, y_cen, 226.750731765,
                      13004.8930993, 103323.763627 ]
        psf_model.update_psf_parameters(psf_params)
        
        sky_model = psf.ConstantBackground()
        sky_model.background_parameters.constant = 1345.0
        
        clean_stamps = psf.subtract_companions_from_psf_stamps(setup, 
                                            reduction_metadata, log, 
                                            ref_star_catalog, psf_idx, 
                                            stamps,stamp_centres,
                                            psf_model,sky_model,diagnostics=True)
    
    logs.close_log(log)


def test_find_psf_companion_stars():
    """Function to test the identification of stars that neighbour a PSF star 
    from the reference catalogue."""
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    log = logs.start_stage_log( cwd, 'test_find_psf_companions' )
    
    log.info(setup.summary())
    
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'reduction_parameters' )

    star_catalog_file = os.path.join(TEST_DATA,'star_catalog.fits')
                            
    ref_star_catalog = catalog_utils.read_ref_star_catalog_file(star_catalog_file)
    
    log.info('Read in catalog of '+str(len(ref_star_catalog))+' stars')
    
    psf_idx = 18
    psf_x = 189.283172607
    psf_y = 9.99084472656
    psf_size = 8.0
    
    stamp_dims = (20,20)
            
    comps_list = psf.find_psf_companion_stars(setup,psf_idx, psf_x, psf_y, 
                                          psf_size, ref_star_catalog,
                                          log, stamp_dims)
            
    assert len(comps_list) > 0
    
    for l in comps_list:
        log.info(repr(l))
    
    image_file = os.path.join(TEST_DATA, 
                            'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
                              
    image = fits.getdata(image_file)
    
    corners = psf.calc_stamp_corners(psf_x, psf_y, stamp_dims[1], stamp_dims[0], 
                                    image.shape[1], image.shape[0],
                                    over_edge=True)
                                    
    stamp = image[corners[2]:corners[3],corners[0]:corners[1]]
    
    log.info('Extracting PSF stamp image')
        
    fig = plt.figure(1)
    
    norm = visualization.ImageNormalize(stamp, \
                interval=visualization.ZScaleInterval())

    plt.imshow(stamp, origin='lower', cmap=plt.cm.viridis, 
               norm=norm)
    
    x = []
    y = []
    for j in range(0,len(comps_list),1):
        x.append(comps_list[j][1])
        y.append(comps_list[j][2])
    
    plt.plot(x,y,'r+')
    
    plt.axis('equal')
    
    plt.xlabel('X pixel')

    plt.ylabel('Y pixel')

    plt.savefig(os.path.join(TEST_DATA,'psf_companion_stars.png'))

    plt.close(1)
    
    logs.close_log(log)


def test_fit_psf_model():
    """Function to test the ability to fit a PSF model to a given stamp
    image, optimizing all parameters
    """
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    log = logs.start_stage_log( cwd, 'test_fit_psf_model' )
    
    log.info(setup.summary())
    
    psf_model_type = 'Moffat2D'
    sky_model_type = 'Constant'
    
    image_file = os.path.join(TEST_DATA, 
                            'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
                              
    image = fits.getdata(image_file)
    
    stamp_centres = np.array([[194.654006958, 180.184967041]])
    
    stamp_dims = (20,20)
        
    stamps = psf.cut_image_stamps(setup, image, stamp_centres, stamp_dims)
    
    fitted_psf = psf.fit_psf_model(setup,log,psf_model_type,sky_model_type,
                                   stamps[0],diagnostics=True)
    
    assert type(fitted_psf) == type(psf.Moffat2D())
    
    log.info('Parameters of fitted PSF model:')
    
    for key in fitted_psf.model:
        
        log.info(key+' = '+str(getattr(fitted_psf.psf_parameters,key)))
        
    logs.close_log(log)

def test_subtract_psf_from_image():
    """Function to test the subtraction of a stellar image from an image"""
    
    image_file = os.path.join(TEST_DATA, 
                            'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
                
                
    image = fits.getdata(image_file)
    
    psf_model = psf.get_psf_object('Moffat2D')
    
    xstar = 194.654006958
    ystar = 180.184967041
    psf_size = 8.0
    x_cen = (psf_size/2.0) + (xstar-int(xstar))
    y_cen = (psf_size/2.0) + (ystar-int(ystar))
    psf_params = [ 5807.59961215, y_cen, x_cen, 7.02930822229, 11.4997891585 ]
    
    psf_model.update_psf_parameters(psf_params)
    
    (residuals,corners) =  psf.subtract_psf_from_image(image,psf_model,xstar,ystar,
                                             psf_size,psf_size)

    assert type(residuals) == type(image)
    
    hdu = fits.PrimaryHDU(residuals)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(image_file.replace('.fits','_res.fits'),
                                 overwrite=True)
    
if __name__ == '__main__':
    
    #test_cut_image_stamps()
    #test_extract_sub_stamp()
    #test_fit_star_existing_model()
    #test_find_psf_companion_stars()
    #test_subtract_companions_from_psf_stamps()
    #test_fit_psf_model()
    #test_build_psf()
    test_subtract_psf_from_image()
    