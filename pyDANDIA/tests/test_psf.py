# -*- coding: utf-8 -*-
"""
Created on Thu Oct 26 15:26:00 2017

@author: rstreet
"""
import pytest
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
import photometry

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
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'star_catalog' )
    
    reduction_metadata.reference_image_path = os.path.join(TEST_DATA,
                                                           'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
    reduction_metadata.background_type = 'constant'
    
    ref_star_catalog = np.zeros([len(reduction_metadata.star_catalog[1]),16])
    
    ref_star_catalog[:,0] = reduction_metadata.star_catalog[1]['star_index'].data  # Index
    ref_star_catalog[:,1] = reduction_metadata.star_catalog[1]['x_pixel'].data  # X
    ref_star_catalog[:,2] = reduction_metadata.star_catalog[1]['y_pixel'].data   # Y
    ref_star_catalog[:,3] = reduction_metadata.star_catalog[1]['RA_J2000'].data       # RA
    ref_star_catalog[:,4] = reduction_metadata.star_catalog[1]['DEC_J2000'].data       # Dec
    ref_star_catalog[:,7] = reduction_metadata.star_catalog[1]['ref_mag'].data  # instrumental mag
    ref_star_catalog[:,8] = reduction_metadata.star_catalog[1]['ref_mag_err'].data # instrumental mag error (null)
    ref_star_catalog[:,15] = reduction_metadata.star_catalog[1]['psf_star'].data  # PSF star switch
    
    log.info('Read in catalog of '+str(len(ref_star_catalog))+' stars')
    
    psf_diameter = 8.0

    ref_image = fits.getdata(reduction_metadata.reference_image_path)
    
    log.info('Loaded reference image')
    
    sky_model = psf.ConstantBackground()
    sky_model.constant = 1345.0
    sky_model.background_parameters.constant = 1345.0

    (psf_model, status) = psf.build_psf(setup, reduction_metadata, log, ref_image, 
                              ref_star_catalog, sky_model, psf_diameter,
                              diagnostics=True)
    
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
    
    Y_image, X_image = np.indices(image.shape)
    
    psf_model = psf.Moffat2D()
    y_cen = 180.184967041
    x_cen = 194.654006958
    psf_radius = 8.0
    psf_params = [ 103301.241291, y_cen, x_cen, 226.750731765,
                  13004.8930993, 103323.763627 ]
    psf_model.update_psf_parameters(psf_params)
    
    sky_model = psf.ConstantBackground()
    sky_model.constant = 1345.0
    sky_model.background_parameters.constant = 1345.0
    sky_bkgd = sky_model.background_model(image.shape,
                                          sky_model.get_parameters())
                                          
    corners = psf.calc_stamp_corners(x_cen, y_cen, psf_radius, psf_radius, 
                                    image.shape[1], image.shape[0],
                                    over_edge=True)
    
    (fitted_model,good_fit) = psf.fit_star_existing_model(setup, image, x_cen, y_cen, 
                                corners, psf_radius, psf_model, sky_bkgd,
                                centroiding=True,
                                diagnostics=True)
                                
    fitted_params = fitted_model.get_parameters()
    
    for i in range(0,5,1):
        print(psf_model.model[i]+': True = '+str(psf_params[i])+', fitted = '+str(fitted_params[i]))
        
        #if i >= 3:        
         #   assert fitted_params[i] == psf_params[i]
    print('Good fit? '+repr(good_fit))
    
    sub_psf_model = psf.get_psf_object('Moffat2D')
            
    pars = fitted_model.get_parameters()
    pars[1] = (psf_radius/2.0) + (y_cen-int(y_cen))
    pars[2] = (psf_radius/2.0) + (x_cen-int(x_cen))
    
    sub_psf_model.update_psf_parameters(pars)
    
    psf_image = psf.model_psf_in_image(image,sub_psf_model,
                            [x_cen,y_cen],
                            diagnostics=True)
    
    residuals = image - psf_image
    
    hdu = fits.PrimaryHDU(residuals)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(os.path.join(TEST_DATA,'ref','sim_star_psf_diff.fits'),
                                     overwrite=True)

def test_fit_existing_psf_stamp():
    """Function to test the function of fitting a pre-existing PSF and sky-
    background model to an image at a stars known location, optimizing just for
    the star's intensity rather than for all parameters."""
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DATA})

    image_file = os.path.join(TEST_DATA, 
                            'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
                              
    image = fits.getdata(image_file)
    
    Y_image, X_image = np.indices(image.shape)
    
    # Calculate the corners of the stamp surrounding the PSF of the star
    # to be subtracted
    y_cen = 180.184967041
    x_cen = 194.654006958
    psf_radius = 8.0
    corners = psf.calc_stamp_corners(x_cen, y_cen, psf_radius, psf_radius, 
                                    image.shape[1], image.shape[0],
                                    over_edge=True)
    x_psf_cen = x_cen - corners[0]
    y_psf_cen = y_cen - corners[2]
    
    # Construct PSF model
    psf_model = psf.Moffat2D()
    psf_params = [ 1.0, y_psf_cen, x_psf_cen, 226.750731765,
                  13004.8930993, 103323.763627 ]
    psf_model.update_psf_parameters(psf_params)
    
    # Construct sky background model
    sky_model = psf.ConstantBackground()
    sky_model.constant = 1600.0
    sky_model.background_parameters.constant = 1345.0
    sky_bkgd = sky_model.background_model(image.shape,
                                          sky_model.get_parameters())
    
    # Extract the stamp section of the image and sky background data
    psf_sky_bkgd = sky_bkgd[corners[2]:corners[3],corners[0]:corners[1]]
    
    psf_image_data = image[corners[2]:corners[3],corners[0]:corners[1]]
    
    Y_data, X_data = np.indices(psf_image_data.shape)
    
    hdu = fits.PrimaryHDU(psf_image_data)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(os.path.join(TEST_DATA,'ref','psf_image_data.fits'),
                                     overwrite=True)
    
    # Fit PSF model to the stamp data
    (fitted_model,good_fit) = psf.fit_existing_psf_stamp(setup, x_psf_cen, y_psf_cen, psf_radius, 
                            psf_model, psf_image_data, psf_sky_bkgd,
                            centroiding=True, diagnostics=False)
    
    fitted_params = fitted_model.get_parameters()
    
    psf_model_image = psf_model.psf_model(Y_data, X_data, fitted_params)
    
    hdu = fits.PrimaryHDU(psf_model_image)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(os.path.join(TEST_DATA,'ref','psf_model_image.fits'),
                                     overwrite=True)
    res = psf_image_data-psf_model_image
    
    y = int(y_psf_cen)
    x = int(x_psf_cen)
    print(x,y,psf_image_data[y,x], psf_model_image[y,x],res[y,x])

    hdu = fits.PrimaryHDU(psf_image_data-psf_model_image)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(os.path.join(TEST_DATA,'ref','psf_diff_image.fits'),
                                     overwrite=True)

    for i in range(0,5,1):
        print(psf_model.model[i]+': True = '+str(psf_params[i])+', fitted = '+str(fitted_params[i]))
        
        #if i >= 3:        
         #   assert fitted_params[i] == psf_params[i]
    print('Good fit? '+repr(good_fit))
    
    sub_psf_model = psf.get_psf_object('Moffat2D')
            
    pars = fitted_model.get_parameters()
    pars[1] = (psf_radius/2.0) + (y_cen-int(y_cen))
    pars[2] = (psf_radius/2.0) + (x_cen-int(x_cen))
    
    sub_psf_model.update_psf_parameters(pars)

    psf_image = psf.model_psf_in_image(image, sub_psf_model,
                            [x_psf_cen,y_psf_cen], 
                            diagnostics=True)
                            
    residuals = image - psf_image
    
    hdu = fits.PrimaryHDU(residuals)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(os.path.join(TEST_DATA,'ref','star_psf_diff.fits'),
                                     overwrite=True)
    
    fig = plt.figure(1,(20,10))
    plt.subplot(1,2,1)
    
    xvalues = np.arange(0,psf_image[y,:].shape[0],1)
    plt.plot(xvalues,(psf_image[y,:]-sky_bkgd[y,:]),'k-',label='Data-sky')
    plt.plot(xvalues,psf_image[y,:],'b-',label='Fitted PSF')
    plt.plot(xvalues,residuals[y,:],'r--',label='Residuals')
    plt.plot(xvalues,[psf_sky_bkgd.mean()]*len(xvalues),'m-.',label='Sky')
    
    plt.xlabel('X [pixel]')
    plt.ylabel('Counts')
    plt.legend()
    
    plt.subplot(1,2,2)
    
    yvalues = np.arange(0,psf_image[:,x].shape[0],1)
    plt.plot(yvalues,(psf_image[:,x]-sky_bkgd[:,x]),'k-',label='Data-sky')
    plt.plot(yvalues,psf_image[:,x],'b-',label='Fitted PSF')
    plt.plot(yvalues,residuals[:,x],'r--',label='Residuals')
    plt.plot(yvalues,[psf_sky_bkgd.mean()]*len(yvalues),'m-.',label='Sky')
    
    plt.xlabel('Y [pixel]')
    plt.ylabel('Counts')
    plt.legend()
    
    plt.savefig(os.path.join(TEST_DATA,'ref','psf_fit_proj.png'),bbox_inches='tight')
    plt.close(1)
    
def test_fit_sim_star_existing_model():
    """Function to test the function of fitting a pre-existing PSF and sky-
    background model to an image at a stars known location, optimizing just for
    the star's intensity rather than for all parameters."""
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    image = np.zeros((8,8))
    image = np.random.rand(image.shape[0],image.shape[1])
    image += 300.0
    
    x_cen = 4.0
    y_cen = 4.0
    psf_params = [ 103301.241291, x_cen, y_cen, 226.750731765,
                  13004.8930993, 103323.763627 ]
    psf_radius = 8.0
    
    sim_star = psf.Moffat2D()
    sim_star.update_psf_parameters(psf_params)
    
    sim_star_image = psf.generate_psf_image('Moffat2D',psf_params,image.shape,psf_radius)
    image += sim_star_image
    
    hdu = fits.PrimaryHDU(image)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(os.path.join(TEST_DATA,'sim_star_psf.fits'),
                                     overwrite=True)
    
    
    psf_fit_params = [ 1.0, x_cen, y_cen, 226.750731765,
                  13004.8930993, 103323.763627 ]
    psf_model = psf.Moffat2D()
    psf_model.update_psf_parameters(psf_fit_params)
    
    sky_model = psf.ConstantBackground()
    sky_model.constant = 300.0
    sky_model.background_parameters.constant = 300.0
    sky_bkgd = sky_model.background_model(image.shape)
    
    (fitted_model,good_fit) = psf.fit_star_existing_model(setup, image, 
                                                        x_cen, y_cen, 
                                                        psf_radius, psf_model, 
                                                        sky_bkgd,
                                                        diagnostics=True)
    
    fitted_params = fitted_model.get_parameters()
    
    for i in range(0,5,1):
        print(psf_model.model[i]+': True = '+str(psf_params[i])+', fitted = '+str(fitted_params[i]))
        
        #assert fitted_params[i] == psf_params[i]

    fitted_star_image = psf.generate_psf_image('Moffat2D',fitted_params,
                                               image.shape,psf_radius)

    hdu = fits.PrimaryHDU(fitted_star_image)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(os.path.join(TEST_DATA,'sim_star_fitted_psf.fits'),
                                     overwrite=True)
                                     
    diff_image = image - fitted_star_image
    
    hdu = fits.PrimaryHDU(diff_image)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(os.path.join(TEST_DATA,'sim_star_psf_diff.fits'),
                                     overwrite=True)
    
    
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
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'star_catalog' )
    
    ref_star_catalog = np.zeros((len(reduction_metadata.star_catalog[1]),9))
    ref_star_catalog[:,0] = reduction_metadata.star_catalog[1]['star_index']
    ref_star_catalog[:,1] = reduction_metadata.star_catalog[1]['x_pixel']
    ref_star_catalog[:,2] = reduction_metadata.star_catalog[1]['y_pixel']
    
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
        psf_diameter = 8.0
        psf_params = [ 103301.241291, x_cen, y_cen, 226.750731765,
                      13004.8930993, 103323.763627 ]
        psf_model.update_psf_parameters(psf_params)
        
        sky_model = psf.ConstantBackground()
        sky_model.background_parameters.constant = 1345.0
        
        clean_stamps = psf.subtract_companions_from_psf_stamps(setup, 
                                            reduction_metadata, log, 
                                            ref_star_catalog, psf_idx, 
                                            stamps,stamp_centres,
                                            psf_model,sky_model,psf_diameter,
                                            diagnostics=True)
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

    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'star_catalog' )
    
    ref_star_catalog = np.zeros((len(reduction_metadata.star_catalog[1]),9))
    ref_star_catalog[:,0] = reduction_metadata.star_catalog[1]['star_index']
    ref_star_catalog[:,1] = reduction_metadata.star_catalog[1]['x_pixel']
    ref_star_catalog[:,2] = reduction_metadata.star_catalog[1]['y_pixel']
        
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

def test_model_psf_in_image():
    """Function to test the subtraction of a stellar image from an image"""
    
    image_file = os.path.join(TEST_DATA, 
                            'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
    
    image = fits.getdata(image_file)
    
    psf_model = psf.get_psf_object('Moffat2D')
    
    xstar = 194.654006958
    ystar = 180.184967041
    psf_size = 8.0
    psf_radius = psf_size/2.0
    x_cen = (psf_size/2.0) + (xstar-int(xstar))
    y_cen = (psf_size/2.0) + (ystar-int(ystar))
    psf_params = [ 5807.59961215, ystar, xstar, 7.02930822229, 11.4997891585 ]
    
    star_data = [xstar, ystar]
    
    psf_model.update_psf_parameters(psf_params)
    
    corners = psf.calc_stamp_corners(xstar, ystar, psf_radius, psf_radius, 
                                    image.shape[1], image.shape[0],
                                    over_edge=True, diagnostics=False)

    psf_image =  psf.model_psf_in_image(image, psf_model, 
                                             star_data, 
                                             diagnostics=True)
    
    assert type(psf_image) == type(image)
    assert psf_image.shape == image.shape
    
    hdu = fits.PrimaryHDU(psf_image)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(image_file.replace('.fits','_psf.fits'),
                                 overwrite=True)

def test_psf_normalization():
    
    psf_model = psf.get_psf_object('Moffat2D')

    params = [1000.0, 10.0, 10.0, 5.0, 10.0]
    
    psf_diameter = 10.0
    
    psf_model.update_psf_parameters(params)
    
    psf_model.normalize_psf(psf_diameter)
    
    Y_data, X_data = np.indices( (int(psf_diameter),int(psf_diameter)) )
    
    (f,ferr) = psf_model.calc_flux(Y_data, X_data)
    
    assert(f == 1.0)

def test_calc_stamp_corners():
    
    image_file = os.path.join(TEST_DATA, 
                              'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
                              
    image = fits.getdata(image_file)
    
    Y_image, X_image = np.indices(image.shape)
    
    # Calculate the corners of the stamp surrounding the PSF of the star
    # to be subtracted
    y_cen = 197.0
    x_cen = 184.0
    psf_radius = 8.0
    corners = psf.calc_stamp_corners(x_cen, y_cen, psf_radius, psf_radius, 
                                    image.shape[1], image.shape[0],
                                    over_edge=True, diagnostics=False)
    
    sub_image_data = image[corners[2]:corners[3],corners[0]:corners[1]]
    
    hdu = fits.PrimaryHDU(sub_image_data)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(os.path.join(TEST_DATA,'ref','corner_image.fits'),
                                     overwrite=True)
    
    assert(corners[0] == 180)
    assert(corners[1] == 188)
    assert(corners[2] == 193)
    assert(corners[3] == 201)

def test_calc_optimized_flux():
    """Function to test the calculation of the optimized flux, given a PSF
    model"""
    
    gain = 1.0
    var_sky = 0.0
    ref_flux = 1.0
    x_star = 5.0
    y_star = 5.0
    
    psf_model = psf.get_psf_object('Moffat2D')
    model_params = [1.0, y_star, x_star, 5.0, 10.0]
    psf_model.update_psf_parameters(model_params)
    
    star_psf = psf.get_psf_object('Moffat2D')
    star_params = [15000.0, y_star, x_star, 5.0, 10.0]
    star_psf.update_psf_parameters(star_params)
    
    psf_diameter = 10.0
    
    Y_data, X_data = np.indices((int(psf_diameter),int(psf_diameter)))
    
    psf_image = psf_model.psf_model(Y_data, X_data, model_params)
    psf_model.normalize_psf(psf_diameter)
    
    star_psf_image = star_psf.psf_model(Y_data, X_data, star_params)
    star_flux = star_psf_image.sum()
    print('Star flux input = '+str(star_flux))
    
    hdu = fits.PrimaryHDU(star_psf_image)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(os.path.join(TEST_DIR,'sim_star_optimize_flux.fits'),
                                     overwrite=True)
    
    (flux, flux_err, Fij) = psf_model.calc_optimized_flux(ref_flux, var_sky, 
                                  y_star, x_star, Y_data, X_data, 
                                  gain, star_psf_image)
                                  
    print('Flux = '+str(flux)+' +/- '+str(flux_err))

    assert flux == pytest.approx(star_flux, 1.0)


def test_calc_optimized_flux_edge():
    """Function to test the calculation of the optimized flux, given a PSF
    model on the edge of the image and hence with an asymmetric stamp."""
    
    gain = 1.0
    var_sky = 0.0
    ref_flux = 1.0
    x_star = 5.0
    y_star = 2.0
    
    psf_model = psf.get_psf_object('Moffat2D')
    model_params = [1.0, y_star, x_star, 5.0, 10.0]
    psf_model.update_psf_parameters(model_params)
    
    star_psf = psf.get_psf_object('Moffat2D')
    star_params = [15000.0, y_star, x_star, 5.0, 10.0]
    star_psf.update_psf_parameters(star_params)
    
    psf_diameter = 10.0
    stamp_width = 10.0
    stamp_height = 5.0
    
    Y_data, X_data = np.indices((int(stamp_height),int(stamp_width)))
    
    psf_image = psf_model.psf_model(Y_data, X_data, model_params)
    psf_model.normalize_psf(psf_diameter)
    
    star_psf_image = star_psf.psf_model(Y_data, X_data, star_params)
    star_flux = star_psf_image.sum()
    print('Star flux input = '+str(star_flux))
    
    hdu = fits.PrimaryHDU(star_psf_image)
    hdulist = fits.HDUList([hdu])
    hdulist.writeto(os.path.join(TEST_DIR,'sim_star_optimize_flux.fits'),
                                     overwrite=True)
    print('Output test image to '+os.path.join(TEST_DIR,'sim_star_optimize_flux.fits'))
    
    (flux, flux_err, Fij) = psf_model.calc_optimized_flux(ref_flux, var_sky, 
                                  y_star, x_star, Y_data, X_data, 
                                  gain, star_psf_image)
                                  
    print('Flux = '+str(flux)+' +/- '+str(flux_err))

    assert flux == pytest.approx(star_flux, 1.0)
    
if __name__ == '__main__':
    
    #test_cut_image_stamps()
    #test_extract_sub_stamp()
    #test_fit_star_existing_model()
    #test_find_psf_companion_stars()
    #test_subtract_companions_from_psf_stamps()
    #test_fit_psf_model()
    test_build_psf()
    #test_model_psf_in_image()
    #test_fit_sim_star_existing_model()
    #test_psf_normalization()
    #test_fit_existing_psf_stamp()
    #test_calc_stamp_corners()
    #test_calc_optimized_flux()
    #test_calc_optimized_flux_edge()