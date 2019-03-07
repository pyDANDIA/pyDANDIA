# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 13:56:46 2017

@author: rstreet
"""

from os import getcwd, path
from sys import path as systempath

class PipelineSetup:
    """Class describing the fundamental parameters needed to identify
    a single-dataset reduction and trigger its reduction
    """
    
    def __init__(self):
        self.red_dir = None
        self.base_dir = None
        self.log_dir = None
        self.phot_db_path = None
        self.pipeline_config_dir = None
        self.software_dir = getcwd()
        self.verbosity = 0

    def summary(self):
        output = 'Reduction directory: '+repr(self.red_dir)+'\n'+\
                'Base directory: '+repr(self.base_dir)+'\n'+\
                'Log directory: '+repr(self.log_dir)+'\n'+\
                'Pipeline configuration directory: '+repr(self.pipeline_config_dir)+'\n'+\
                'Photometric database path: '+repr(self.phot_db_path)+'\n'+\
                'Software directory: '+repr(self.software_dir)+'\n'+\
                'Verbosity level: '+str(self.verbosity)
        return output

def pipeline_setup(params):
    """Function to acquire the necessary commandline arguments to run
    pyDANDIA in pipeline mode."""
    
    setup = PipelineSetup()
    
    if 'red_dir' in params.keys():
        setup.red_dir = params['red_dir']
    
        setup.phot_db_path = path.join(params['red_dir'],'..','phot.db')
    
    if 'base_dir' in params.keys():
        setup.base_dir = params['base_dir']
        
        setup.phot_db_path = path.join(params['base_dir'],'phot.db')
        
    for key in ['log_dir', 'pipeline_config_dir', 'software_dir', 'verbosity']:
        
        if key in params.keys():
        
            if key == 'log_dir':
                
                setup.log_dir = params[key]
            
            elif key == 'pipeline_config_dir':
                
                setup.pipeline_config_dir = params[key]
            
            elif key == 'software_dir':
                
                setup.software_dir = params[key]
            
            elif key == 'verbosity':
            
                setup.verbosity = params[key]
        
        else:
            
            if key == 'log_dir':
                
                setup.log_dir = path.join(setup.red_dir,'..','logs')
            
            elif key == 'pipeline_config_dir':
                
                setup.pipeline_config_dir = path.join(setup.red_dir,
                                                              '..','config')
            
            elif key == 'software_dir':
                
                setup.software_dir = getcwd()
            
            elif key == 'verbosity':
            
                setup.verbosity = 2
            
    return setup
