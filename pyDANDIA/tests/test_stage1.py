import os
from os import getcwd, path
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import starfind
import sys
import metadata
import logs
import stage1
import pipeline_setup

TEST_DATA = path.join(cwd,'data/proc/ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')

def test_stage1():
    """ Function to test the performance of stage1"""
    
    params = {}
    
    params['red_dir'] = TEST_DATA
    params['verbosity'] = 2
    params['log_dir'] = os.path.join(params['red_dir'],'..','logs')
    params['pipeline_config_dir'] = os.path.join(params['red_dir'],'..','configs')
    params['software_dir'] = os.getcwd()
    
    test_setup = pipeline_setup.pipeline_setup(params)
    (status, report) = stage1.run_stage1(test_setup)
    
    assert status == 'OK'
    assert report == 'Completed successfully'

if __name__ == '__main__':
    test_stage1()
    
