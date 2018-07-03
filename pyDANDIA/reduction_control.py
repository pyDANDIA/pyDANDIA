# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:01:19 2017

@author: rstreet
"""
from os import getcwd, path, remove
from sys import argv, exit
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import pipeline_setup
import stage0
import stage1
import stage2
import stage3
import stage4
import stage5
#import stage6
import logs

def reduction_control():
    """Main driver function for the pyDANDIA pipelined reduction of an 
    individual dataset.  
    
    A single dataset is defined as a set of images taken with a consistent
    instrument configuration and filter of a single sky pointing.  
    
    Input parameters:
        dataset_red_dir   str   Full path to the reduction directory for this
                                dataset
    """

    reduction_version = 'reduction_control v0.2'

    setup = get_args()
    
    log = logs.start_pipeline_log(setup.red_dir, 'reduction_control',
                                  version=reduction_version)

    (status,report,meta_data) = stage0.run_stage0(setup)
    log.info('Completed stage 0 with status '+repr(status)+': '+report)
    
    (status, report) = stage1.run_stage1(setup)
    log.info('Completed stage 1 with status '+repr(status)+': '+report)

    (status, report) = stage2.run_stage2(setup)
    log.info('Completed stage 2 with status '+repr(status)+': '+report)
    
    (status, report) = stage3.run_stage3(setup)
    log.info('Completed stage 3 with status '+repr(status)+': '+report)
    
    (status, report) = stage4.run_stage4(setup)
    log.info('Completed stage 4 with status '+repr(status)+': '+report)
    
    (status, report) = stage5.run_stage5(setup)
    log.info('Completed stage 5 with status '+repr(status)+': '+report)
 
# Code deactivated until stage 6 is fully integrated with pipeline   
#    (status, report) = stage6.run_stage6(setup)
#    log.info('Completed stage 6 with status '+repr(status)+': '+report)
    
    logs.close_log(log)


def get_args():
    """Function to obtain the command line arguments necessary to run a 
    single-dataset reduction."""
    
    helptext = """
                    pyDANDIA Reduction Control
    
    Main driver program to run pyDANDIA in pipeline mode for a single dataset. 
    
    Command and options:
    > python reduction_control.py red_dir_path [-v N ]
    
    where red_dir_path is the path to a dataset's reduction directory
    
    The -v flag controls the verbosity of the pipeline logging output.  Values 
    N can be:
    -v 0 [Default] Essential logging output only, written to log file. 
    -v 1           Detailed logging output, written to log file.
    -v 2           Detailed logging output, written to screen and to log file.

    To display information on options:    
    > python reduction_control.py -help
    """
        
    if '-help' in argv:
        print(helptext)
        exit()
    
    params = {}
    
    if len(argv) == 1:
        
        params['red_dir'] = raw_input('Please enter the path to the datasets reduction directory: ')
    
    else:
        
        params['red_dir'] = argv[1]
    
    if '-v' in argv:
        
        idx = argv.index('-v')
        
        if len(argv) >= idx + 1:
            
            params['verbosity'] = int(argv[idx+1])
        
    params['log_dir'] = path.join(params['red_dir'],'..','logs')
    params['pipeline_config_dir'] = path.join(params['red_dir'],'..','config')
    params['software_dir'] = getcwd()
    
    setup= pipeline_setup.pipeline_setup(params)
    
    return setup
    
    
    
if __name__ == '__main__':
    reduction_control()
