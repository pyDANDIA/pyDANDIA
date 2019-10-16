# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:34:27 2019

@author: rstreet
"""

from os import getcwd, path, remove
from sys import argv
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import pipeline_setup
import glob
import subprocess
import logs
import dataset_utils 
import reduction_control

def run_reductions():
    """Function to drive multiple, parallelized reductions of several datasets"""
    
    params = get_args()
    
    data = dataset_utils.DataCollection()
    
    data.get_datasets_for_reduction(params['datasets_file'])
    
    log = logs.start_pipeline_log(data.red_dir, 'prepare_datasets')
    
    log.info(data.summary())
    print(data.summary())
    
    setup = dataset_utils.build_pipeline_setup(data)
    
    check_sanity(setup,data,log)
        
    run_parallel_data_preparations(setup, data, log)
    
    logs.close_log(log)
    
def get_args(debug=False):
    """Function to acquire the necessary commandline arguments to run
    pyDANDIA in pipeline mode."""
    
    params = {}
    
    if len(argv) == 1:
        
        params['datasets_file'] = input('Please enter the path to the file listing datasets to be reduced: ')
        
    elif debug == False and len(argv) > 1:

        params['datasets_file'] = argv[1]
        
    return params

def check_sanity(setup,data,log):
    
    if path.isdir(setup.base_dir) == False:
        raise IOError('ERROR: Cannot find reduction base directory '+setup.base_dir)
        exit()
    else:
        log.info('Reduction base directory confirmed')
        
    if path.isdir(data.red_dir) == None:
        raise IOError('ERROR: Top-level reduction directory not set')
        exit()
    else:
        log.info('Reduction directory configured')
    
    if len(data.data_list) == 0:
        raise IOError('ERROR: No datasets to be reduced')

def run_parallel_data_preparations(setup, data, log):
    """Function to execute multiple reductions of separate datasets 
    in parallel"""
    
    log.info('Starting reductions:')
    
    for data_dir in data.data_list:
        
        dataset_dir = path.join(data.red_dir,data_dir)
        
        pid = trigger_reduction(setup,dataset_dir,'data_preparation',debug=False)
        
        log.info(' -> Dataset '+path.basename(dataset_dir)+\
                ' reduction PID '+str(pid))

def trigger_reduction(setup,dataset_dir,red_mode,debug=False):
    """Function to spawn a child process to run the reduction of a 
    single dataset.
    
    Inputs:
        setup       PipelineSetup object
        dataset_dir   str    Path to dataset red_dir
    """
    
    if debug == False:
        
        command = path.join(setup.software_dir,'reduction_control.py')
    
    else:
        
        if 'tests' in setup.software_dir:
            
            command = path.join(setup.software_dir,'counter.py')
        
        else:
            
            command = path.join(setup.software_dir,'tests','counter.py')
            
    args = ['python', command, dataset_dir, red_mode]
    
    p = subprocess.Popen(args, stdout=subprocess.PIPE)
    
    return p.pid
    

if __name__ == '__main__':
    
    run_reductions()
    