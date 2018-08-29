# -*- coding: utf-8 -*-
"""
Created on Wed Aug 29 13:10:43 2018

@author: rstreet
"""
import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import logs
import pipeline_setup
import bad_pixel_mask
import numpy as np
import glob

params = {'red_dir': os.path.join(cwd, 'data', 'proc', 
                                   'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip'), 
          'log_dir': os.path.join(cwd, 'data', 'proc', 
                                   'logs'),
          'pipeline_config_dir': os.path.join(cwd, 'data', 'proc', 
                                   'config'),
          'software_dir': os.path.join(cwd, '..'),
          'verbosity': 2
         }
            
def test_read_mask():
    """Function to check that a standard FITS format bad pixel mask can be 
    read in correctly"""
    
    file_path = raw_input('Please enter the path to a bad pixel mask file: ')
    
    bpm = bad_pixel_mask.BadPixelMask()
    
    bpm.read_mask(file_path)
    
    assert type(bpm.data) == type(np.zeros([1]))
    assert bpm.data.shape[0] > 4000
    assert bpm.data.shape[1] > 4000

def test_load_mask():
    """Function to verify that the pipeline can identify and load the most 
    recent example of a bad pixel file from the given configuration directory
    """
    
    setup = pipeline_setup.pipeline_setup(params)
    
    setup.pipeline_config_dir = raw_input('Please enter the path to the config directory: ')
    camera = raw_input('Please enter the camera to search for: ')
    
    bpm = bad_pixel_mask.BadPixelMask()
    
    bpm.load_latest_mask(camera,setup)

    bpm_list = glob.glob(os.path.join(setup.pipeline_config_dir,'bpm*'+camera+'*fits'))
    
    date_list = []
    
    for f in bpm_list:
        
        date_list.append(str(os.path.basename(f)).replace('.fits','').split('_')[2])
    
    idx = (np.array(date_list)).argsort()
    
    latest_date = str(os.path.basename(bpm_list[idx[-1]])).replace('.fits','').split('_')[2]
    
    assert bpm.camera == camera
    assert bpm.dateobs == latest_date
    assert type(bpm.data) == type(np.zeros([1]))
    assert bpm.data.shape[0] > 4000
    assert bpm.data.shape[1] > 4000
    
if __name__ == '__main__':
    
    #test_read_mask()
    test_load_mask()