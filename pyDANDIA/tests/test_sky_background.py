# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 15:14:34 2017

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
import sky_background
import catalog_utils
from astropy.io import fits

TEST_DIR = os.path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')

def test_gradient_background():
    """Function to test the generation of a background sky model with a 
    2D gradient function
    """
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})

    Y_data, X_data = np.indices((400,400))

    params = { 'a0': 42.0, 'a1': 1.5, 'a2': 3.0 }

    model = psf.GradientBackground()
    
    model.update_background_parameters(params)
    
    sky_model = model.background_model(Y_data, X_data, params)
    
    fig = plt.figure(1)
    
    plt.imshow(sky_model, origin='lower', cmap=plt.cm.viridis)
            
    plt.xlabel('X pixel')

    plt.ylabel('Y pixel')

    plt.savefig(os.path.join(setup.red_dir,'sky_background_model.png'))

    plt.close(1)
    
    assert (os.path.isfile(os.path.join(setup.red_dir,'sky_background_model.png'))==True)


def test_sky_model():
    """Function to test the function used to compute the sky background 2D 
    image during the background fitting process."""

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    params = { 'background_type': 'constant', 
          'nx': 400, 'ny': 400,
          'constant': 42.0 
          }

    sky_model1 = sky_background.generate_sky_model_image(params)
    
    params = { 'background_type': 'gradient', 
          'nx': 400, 'ny': 400,
          'a0': 42.0, 'a1': 1.2, 'a2': 3.0
          }

    sky_model2 = sky_background.generate_sky_model_image(params)

    fig = plt.figure(2)
    
    plt.subplot(2, 1, 1)
    
    plt.imshow(sky_model1, origin='lower', cmap=plt.cm.viridis)
    
    plt.title('Constant background model')
    
    plt.subplot(2, 1, 2)
    
    plt.imshow(sky_model2, origin='lower', cmap=plt.cm.viridis)
    
    plt.title('2D gradient background model')
    
    plt.savefig(os.path.join(setup.red_dir,'sky_background_model_test.png'))

    plt.close(2)

    assert (os.path.isfile(os.path.join(setup.red_dir,'sky_background_model_test.png'))==True)

def test_error_sky_fit_function():
    """Function to test the sky-fitting procedure's error function"""

    params = { 'background_type': 'gradient', 
          'nx': 400, 'ny': 400,
          'a0': 42.0, 'a1': 1.2, 'a2': 3.0
          }

    par_list = [ params['a0'], params['a1'], params['a2'] ]
    
    sky_model1 = sky_background.generate_sky_model_image(params)
    
    test_sky = np.zeros(sky_model1.shape)
    
    sky_error = sky_background.error_sky_fit_function(par_list,sky_model1, 
                                    params['background_type'])
    
    assert len(sky_error.shape) == 1
    assert sky_error.min() == 0.0
    assert sky_error.max() == 0.0
    
    params = { 'background_type': 'gradient', 
          'nx': 400, 'ny': 400,
          'a0': -42.0, 'a1': -0.1, 'a2': 0.02
          }
    
    sky_model2 = sky_background.generate_sky_model_image(params)

    sky_error = sky_background.error_sky_fit_function(par_list,sky_model2, 
                                    params['background_type'])

    assert sky_error.min() < 0.0
    assert sky_error.max() < 0.0

def test_fit_model_sky_background():
    """Function to test the fitting procedure of a model to the sky background
    of an image."""
    
    background_type = 'gradient'
    
    image_params = { 'background_type': background_type, 
          'nx': 400, 'ny': 400,
          'a0': 42.0, 'a1': 1.2, 'a2': 3.0
          }

    model_params = { 'background_type': background_type, 
          'nx': 400, 'ny': 400,
          'a0': 0.0, 'a1': 0.0, 'a2': 0.0
          }
          
    image = sky_background.generate_sky_model_image(image_params)
    
    sky_model = sky_background.generate_sky_model(model_params)
    
    sky_fit = sky_background.fit_sky_background(image,sky_model,background_type)
    
    for i, p in enumerate([ 'a0', 'a1', 'a2' ]):
        
        assert sky_fit[0][i] == image_params[p]

def test_model_sky_background():
    """Function to test the fitting of a sky background model to a masked
    real star image."""
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    log = logs.start_stage_log( cwd, 'test_sky_background' )
    
    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'reduction_parameters' )
    reduction_metadata.load_a_layer_from_file( setup.red_dir, 
                                              'pyDANDIA_metadata.fits', 
                                              'images_stats' )

    log.info('Read metadata')

    # Need to check where these parameters come from
    reduction_metadata.reference_image_path = os.path.join(cwd,'data',
                                                           'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
    reduction_metadata.background_type = 'constant'
    
    ref_star_catalog_file = os.path.join(cwd,'data','star_catalog.fits')
    
    ref_star_catalog = catalog_utils.read_ref_star_catalog_file(ref_star_catalog_file)
    
    log.info('Read reference image star catalog from '+ref_star_catalog_file)
        
    sky_model = sky_background.model_sky_background(setup,reduction_metadata,
                                                  log,ref_star_catalog)
    
    log.info('Fit image sky background with '+\
            reduction_metadata.background_type+' model, parameters:')
    
    for key in sky_model.model:
        
        log.info(key+' = '+str(getattr(sky_model.background_parameters,key)))
    
    logs.close_log(log)

def test_mask_saturated_pixels():
    """Function to test the masking of saturated pixels in a FITS image."""
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    log = logs.start_stage_log( cwd, 'test_sky_background' )
    
    test_image_file = os.path.join(cwd,'data',
                                       'lsc1m005-fl15-20170701-0144-e91_cropped.fits')
    
    hdulist = fits.open(test_image_file)
    
    saturation_value = hdulist[0].header['SATURATE']

    image = hdulist[0].data
    
    masked_image = sky_background.mask_saturated_pixels(setup,image,
                                                        saturation_value,log)

    fig = plt.figure(4)
    
    plt.imshow(masked_image, origin='lower', cmap=plt.cm.viridis)
    
    plt.title('Masked image')
    
    plt.savefig(os.path.join(cwd, 'data','masked_image_test.png'))

    plt.close(4)
    
    assert (os.path.isfile(os.path.join(cwd, 'data','masked_image_test.png')))

    logs.close_log(log)

if __name__ == '__main__':
    
    test_gradient_background()
    test_sky_model()
    test_error_sky_fit_function()
    test_fit_model_sky_background()
    test_mask_saturated_pixels()
    test_model_sky_background()
    