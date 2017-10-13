# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 14:09:50 2017

@author: rstreet
"""

from os import getcwd, path
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import pipeline_setup

def test_pipeline_setup():
    """Function to set the essential parameters to run the pyDANDIA pipeline 
    can be set correctly
    """
    
    params = {'red_dir': path.join(cwd, 'data', 'proc', 
                                   'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip'), 
              'log_dir': path.join(cwd, 'data', 'proc', 
                                   'logs'),
              'pipeline_config_dir': path.join(cwd, 'data', 'proc', 
                                   'configs'),
              'software_dir': path.join(cwd, '..'),
              'verbosity': 2
            }
    
    setup = pipeline_setup.pipeline_setup(params)
    
    test_setup = pipeline_setup.PipelineSetup()
    
    assert type(setup) == type(test_setup)