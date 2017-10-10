# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 08:50:19 2017

@author: rstreet
"""

from os import getcwd, path, remove
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
from shutil import copyfile 
import config_utils

TEST_DATA = path.join(cwd,'data')

def test_read_config():
    """Function to unit test the reading of the pipeline's JSON config files"""
    
    config_file_path = '../../Config/config.json'
    
    config = config_utils.read_config(config_file_path)
    
    assert type(config) == type({'a':1, 'b':2})
    assert 'proc_data' in config.keys()
    
    config_file_path = '../../Config/inst_config.json'
    
    config = config_utils.read_config(config_file_path)
    
    assert type(config) == type({'a':1, 'b':2})
    assert 'instrid' in config.keys()
    
    
def test_set_config_value():

    test_key = 'proc_data'
    new_value = '/Users/test/path'    
    
    config_file_path = path.join(TEST_DATA,'config.json')
    copyfile('../../Config/config.json',config_file_path)
    
    init_config = config_utils.read_config(config_file_path)
    
    status = config_utils.set_config_value(config_file_path, test_key, new_value)
    
    updated_config = config_utils.read_config(config_file_path)
        
    assert updated_config[test_key]['value'] == new_value
    
    remove(config_file_path)