# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:16:20 2017

@author: rstreet
"""

from os import getcwd, path, remove
from sys import argv
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import pipeline_setup
import pipeline_control
import logs

TEST_DATA = path.join(cwd,'data')

VERSION = 'test_pipeline_control v0.1'

params = {'red_dir': path.join(cwd, 'data', 'proc', 
                                   'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip'), 
              'log_dir': path.join(cwd, 'data', 'proc', 
                                   'logs'),
              'pipeline_config_dir': path.join(cwd, 'data', 'proc', 
                                   'configs'),
              'software_dir': path.join(cwd, '..'),
              'verbosity': 2
            }
            
def test_get_datasets():
    """Function to test the function which identifies which datasets
    should be reduced.
    """

    setup = pipeline_setup.pipeline_setup(params)
    
    log = logs.start_pipeline_log(setup.log_dir, 'test_pipeline_control', 
                               version=VERSION)
    
    datasets = pipeline_control.get_datasets_for_reduction(setup,log)
    
    assert type(datasets) == type(['a','b','c'])

    logs.close_log(log)

def test_trigger_reduction():
    """Function to test the spawning of reduction processes"""
    
    setup = pipeline_setup.pipeline_setup(params)

    log = logs.start_pipeline_log(setup.log_dir, 'test_pipeline_control', 
                               version=VERSION)

    dataset_dir = TEST_DATA
    
    pid = pipeline_control.trigger_reduction(setup,dataset_dir,debug=True)
    
    assert type(pid) == type(1)
    
    logs.close_log(log)
