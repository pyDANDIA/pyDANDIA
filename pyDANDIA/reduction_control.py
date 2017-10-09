# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:01:19 2017

@author: rstreet
"""
from os import getcwd, path, remove
from sys import argv
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import stage0
import logs

VERSION = 'reduction_control v0.1'

def reduction_control():
    """Main driver function for the pyDANDIA pipelined reduction of an 
    individual dataset.  
    
    A single dataset is defined as a set of images taken with a consistent
    instrument configuration and filter of a single sky pointing.  
    
    Input parameters:
        dataset_red_dir   str   Full path to the reduction directory for this
                                dataset
    """

    setup = get_args()
    
    
    log = logs.start_pipeline_log(setup.red_dir, 'reduction_control', 
                               version=VERSION)

    (status,meta_data) = stage0.run_stage0(setup)
    log.info('Completed stage 0 with status '+repr(status))
    
    
    logs.close_log(log)

class ReductionSetup:
    """Class describing the fundamental parameters needed to identify
    a single-dataset reduction and trigger its reduction
    """
    
    def __init__(self):
        self.red_dir = None
        self.log_dir = None
        self.pipeline_config_dir = None
        self.software_dir = getcwd()
        
    def summary(self):
        output = 'Reduction directory: '+repr(self.red_dir)+'\n'+\
                'Log directory: '+repr(self.log_dir)+'\n'+\
                'Pipeline configuration directory: '+repr(self.pipeline_config_dir)+'\n'+\
                'Software directory: '+repr(self.software_dir)
        return output
        
def get_args():
    """Function to obtain the command line arguments necessary to run a 
    single-dataset reduction."""
    
    setup = ReductionSetup()
    
    if len(argv) == 1:
        
        setup.red_dir = raw_input('Please enter the path to the datasets reduction directory: ')
    
    else:
    
        setup.red_dir = argv[1]
    
    setup.log_dir = path.join(setup.red_dir,'..','logs')
    setup.pipeline_config_dir = path.join(setup.red_dir,'..','configs')
    setup.software_dir = getcwd()
    
    return setup
    
    
    
if __name__ == '__main__':
    reduction_control()