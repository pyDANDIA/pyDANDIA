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

TEST_DIR = os.path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')

def test_gradient_background():
    """Function to test the generation of a background sky model with a 
    2D gradient function
    """
    
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

    Y_data, X_data = np.indices((400,400))

    params = { 'a0': 42.0, 'a1': 1.5, 'a2': 3.0 }

    sky_model = psf.GradientBackground()
    
    sky_model.update_background_parameters(params)
    
    sky_model = sky_model.background_model(Y_data, X_data, params)
    
    fig = plt.figure(1)
    
    plt.imshow(sky_model, origin='lower', cmap=plt.cm.viridis)
            
    plt.xlabel('X pixel')

    plt.ylabel('Y pixel')

    plt.savefig(os.path.join(setup.red_dir,'sky_background_model.png'))

    plt.close(1)
    
    log.info('Output simultated background sky model to '+\
            os.path.join(setup.red_dir,'sky_background_model.png'))

    assert (os.path.isfile(os.path.join(setup.red_dir,'sky_background_model.png'))==True)

    logs.close_log(log)

def test_sky_fit_function():
    """Function to test the function used to compute the sky background 2D 
    image during the background fitting process."""

    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    params = { 'background_type': 'constant', 
          'nx': 400, 'ny': 400,
          'constant': 42.0 
          }

    sky_model1 = sky_background.sky_fit_function(params)
    
    params = { 'background_type': 'gradient', 
          'nx': 400, 'ny': 400,
          'a0': 42.0, 'a1': 1.2, 'a2': 3.0
          }

    sky_model2 = sky_background.sky_fit_function(params)

    fig, axarr = plt.subplots(2, sharex=True)
    
    axarr[0].imshow(sky_model1, origin='lower', cmap=plt.cm.viridis)
    
    axarr[0].set_title('Constant background model')
            
    axarr[1].imshow(sky_model2, origin='lower', cmap=plt.cm.viridis)
    
    axarr[1].set_title('2D gradient background model')
    
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
    
    sky_model1 = sky_background.sky_fit_function(params)
    
    test_sky = np.zeros(sky_model1.shape)
    
    sky_error = sky_background.error_sky_fit_function(sky_model1, 
                                    par_list, 
                                    background_type=params['background_type'])
    
    delta = sky_error - test_sky
    assert delta.min() == 0.0
    assert delta.max() == 0.0
    
    params = { 'background_type': 'gradient', 
          'nx': 400, 'ny': 400,
          'a0': -42.0, 'a1': -0.1, 'a2': 0.02
          }
    
    sky_model2 = sky_background.sky_fit_function(params)

    sky_error = sky_background.error_sky_fit_function(sky_model2, 
                                    par_list, 
                                    background_type=params['background_type'])

    assert sky_error.min() < 0.0
    assert sky_error.max() < 0.0
    
    
if __name__ == '__main__':
    
    test_gradient_background()
    test_sky_fit_function()
    test_error_sky_fit_function()