# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:44:15 2018

@author: rstreet
"""


import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import logs
import pipeline_setup
import metadata
import reduction_control
import stage0

TEST_DATA = os.path.join(cwd,'data')

VERSION = 'test_reduction_control v0.1'

params = {'red_dir': os.path.join(cwd, 'data', 'proc', 
                                   'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip'), 
              'log_dir': os.path.join(cwd, 'data', 'proc', 
                                   'logs'),
              'pipeline_config_dir': os.path.join(cwd, 'data', 'proc', 
                                   'config'),
              'software_dir': os.path.join(cwd, '..'),
              'verbosity': 2
            }

def prepare_test_directory(setup):
    """Function to clean the test data directory of old data products"""
    
    stage_log = os.path.join(setup.red_dir,'stage0.log')
    metadata = os.path.join(setup.red_dir,'pyDANDIA_metadata.fits')
    
    if os.path.isfile(stage_log):
        os.remove(stage_log)
   
    return stage_log, metadata
    
def test_trigger_stage_subprocess():
    """Function to test the running of a stage as a subprocess"""
    
    setup = pipeline_setup.pipeline_setup(params)
    
    log = logs.start_pipeline_log(setup.log_dir, 'test_reduction_control', 
                               version=VERSION)
    
    (stage_log,metadata) = prepare_test_directory(setup)
        
    process = reduction_control.trigger_stage_subprocess('stage0',setup,log,
                                                         wait=True)
    
    assert type(process.pid) == type(0)
    assert os.path.isfile(stage_log) == True
    assert os.path.isfile(metadata) == True

    logs.close_log(log)
    
def test_execute_stage():
    """Function to test the execution of a stage direct from a function"""
    
    setup = pipeline_setup.pipeline_setup(params)
    
    log = logs.start_pipeline_log(setup.log_dir, 'test_reduction_control', 
                               version=VERSION)
    
    (stage_log,metadata) = prepare_test_directory(setup)
    
    status = 'OK'
    status = reduction_control.execute_stage(stage0.run_stage0, 'stage 0', 
                                             setup, status, log)

    assert os.path.isfile(stage_log) == True
    assert os.path.isfile(metadata) == True
    
    logs.close_log(log)
    
if __name__ == '__main__':
    
    test_trigger_stage_subprocess()
    test_execute_stage()
    