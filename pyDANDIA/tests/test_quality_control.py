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
    
    
if __name__ == '__main__':
    
    test_verify_stage0_output()
    