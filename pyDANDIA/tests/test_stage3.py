# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:58:40 2017

@author: rstreet
"""

import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import logs
import pipeline_setup
import metadata
import stage3

TEST_DIR = os.path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')

def test_run_stage3():
    """Function to test the execution of Stage 3 of the pipeline, end-to-end"""
    
    setup = pipeline_setup.pipeline_setup({'red_dir': TEST_DIR})
    
    stage3.run_stage3(setup)
    
    
if __name__ == '__main__':
    
    test_run_stage3()