# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 13:48:25 2018

@author: rstreet
"""

import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import logs
import pipeline_setup
import metadata
import quality_control
import stage0
import stage1
from astropy.io import fits
import numpy as np

params = {'red_dir': os.path.join(cwd, 'data', 'proc',
                                   'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip'),
              'log_dir': os.path.join(cwd, 'data', 'proc',
                                   'logs'),
              'pipeline_config_dir': os.path.join(cwd, 'data', 'proc',
                                   'config'),
              'software_dir': os.path.join(cwd, '..'),
              'verbosity': 2
            }

def test_verify_stage0_output():
    """Function to test the verification of stage 0 data products"""

    setup = pipeline_setup.pipeline_setup(params)

    log = logs.start_pipeline_log(setup.log_dir, 'test_quality_control')

    stage_log = os.path.join(setup.red_dir,'stage0.log')

    if os.path.isfile(stage_log) == False:

        (fstatus, freport, reduction_metadata) = stage0.run_stage0(setup)

    (status, report) = quality_control.verify_stage0_output(setup,log)

    assert 'OK' in status
    assert 'success' in report

    logs.close_log(log)

def test_assess_image():
    """Function to test the assessment of an image based on its measured
    photometric properties"""

    setup = pipeline_setup.pipeline_setup(params)

    log = logs.start_pipeline_log(setup.log_dir, 'test_quality_control')

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'reduction_parameters' )

    image_params = {'nstars': 10000,
              'sky': 450.0,
              'fwhm_x': 1.2,
              'fwhm_y': 1.25
              }

    (flag, report) = quality_control.assess_image(reduction_metadata,
                                                    image_params,log)

    assert flag == 1
    assert 'OK' in report

    image_params = {'nstars': 0,
              'sky': 18000.0,
              'fwhm_x': 1.2,
              'fwhm_y': 1.25
              }

    (flag,report) = quality_control.assess_image(reduction_metadata,
                                                    image_params,log)

    assert flag == 0
    assert 'Sky' in report

    image_params = {'nstars': 0,
              'sky': 1200.0,
              'fwhm_x': 3.0,
              'fwhm_y': 3.2
              }

    (flag,report) = quality_control.assess_image(reduction_metadata,
                                                    image_params,log)

    assert flag == 0
    assert 'FWHM' in report

    logs.close_log(log)


def test_verify_stage1_output():
    """Function to test the verification of stage 0 data products"""

    setup = pipeline_setup.pipeline_setup(params)

    log = logs.start_pipeline_log(setup.log_dir, 'test_quality_control')

    stage_log = os.path.join(setup.red_dir,'stage0.log')

    if os.path.isfile(stage_log) == False:

        (fstatus, freport, reduction_metadata) = stage0.run_stage0(setup)
        (fstatus, freport) = stage1.run_stage1(setup)

    (status, report) = quality_control.verify_stage1_output(setup,log)

    assert 'OK' in status
    assert 'success' in report

    logs.close_log(log)

def test_verify_mask_statistics():
    """Function to test the verification of image bad pixel masking"""

    setup = pipeline_setup.pipeline_setup(params)

    log = logs.start_pipeline_log(setup.log_dir, 'test_quality_control')

    test_image = 'lsc1m005-fl15-20170418-0131-e91_cropped.fits'
    hdu = fits.open(os.path.join(params['red_dir'], 'data', test_image), memmap=True)
    mask_data = np.array(hdu[-1].data, dtype=float)

    mask_status = quality_control.verify_mask_statistics(mask_data, log)

    logs.close_log(log)

    assert type(mask_status) == type(True)
    
if __name__ == '__main__':

    #test_verify_stage0_output()
    #test_assess_image()
    #test_verify_stage1_output()
    test_verify_mask_statistics()
