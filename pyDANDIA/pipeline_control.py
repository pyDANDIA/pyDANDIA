# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:09:55 2017

@author: rstreet
"""
from os import getcwd, path, remove
from sys import argv
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import glob
import subprocess
import logs

VERSION = 'pipeline_control v0.1'

def pipeline_control():
    """Main driver program controlling the reduction of multiple datasets 
    with pyDANDIA.
    """
    
    setup = pipeline_setup()

    log = logs.start_pipeline_log(setup.log_dir, 'pipeline_control', 
                               version=VERSION)

    datasets = get_datasets_for_reduction(setup,log)
    
    run_reductions(setup,log,datasets)

    logs.close_log(log)
    
    
class PipelineSetup:
    """Class describing the fundamental parameters needed to identify
    a single-dataset reduction and trigger its reduction
    """
    
    def __init__(self):
        self.base_dir = None
        self.log_dir = None
        self.pipeline_config_dir = None
        self.software_dir = getcwd()


def pipeline_setup(debug=False):
    """Function to acquire the necessary commandline arguments to run
    pyDANDIA in pipeline mode."""
    
    setup = PipelineSetup()
    
    if debug == True:
        
        setup.base_dir = cwd
        
    elif debug == False and len(argv) == 1:
        
        setup.base_dir = raw_input('Please enter the path to the base directory: ')
        
    elif debug == False and len(argv) > 1:

        setup.base_dir = argv[1]
    
    if path.isdir(setup.base_dir) == False:
        
        print('ERROR: Cannot find reduction base directory '+setup.base_dir)
    
    setup.log_dir = path.join(setup.base_dir,'logs')
    setup.pipeline_config_dir = path.join(setup.base_dir,'configs')
    
    return setup
    
    
def get_datasets_for_reduction(setup,log):
    """Function to compose a list of the datasets to be reduced.
    
    Options:
    1) If a file reduce_datasets.txt exists within the proc/configs directory
    this will be read.  This file should contain a list (one per line) of the 
    reduction sub-directory names to be reduced, without paths.  
    
    2) If no reduce_datasets.txt file is found the code returns a list of 
    all of the dataset sub-directories found in the /proc directory.
    """
    
    datasets_file = path.join(setup.pipeline_config_dir,'reduce_datasets.txt')
    
    if path.isfile(datasets_file) == True:

        log.info('Found a reduce_datasets instruction file')

        file_lines = open(datasets_file).readlines()
        
        datasets = []
        
        log.info('Going to reduce the following datasets:')
        
        for line in file_lines:
            
            datasets.append(line.replace('\n',''))
            
            log.info(datasets[-1])
            
    else:
        
        log.info('No instruction file found, going to reduce all datasets')
        
        dir_list = glob.glob(path.join(setup.base_dir,'*'))
        
        datasets = []
        
        for item in dir_list:
            
            if 'logs' not in item and 'configs' not in item:
                
                datasets.append(path.basename(item))
                
                log.info(datasets[-1])
            
    return datasets


def run_reductions(setup,log,datasets):
    """Function to trigger the reduction of one or more datasets.
    
    Inputs: 
        setup       PipelineSetup object
        datasets    list                    Dataset red_dir names
    """

    log.info('Starting reductions:')
    
    for data_dir in datasets:
        
        dataset_dir = path.join(setup.base_dir,data_dir)
        
        pid = trigger_reduction(setup,dataset_dir,debug=False)
        
        log.info(' -> Dataset '+path.basename(dataset_dir)+\
                ' reduction PID '+str(pid))

def trigger_reduction(setup,dataset_dir,debug=False):
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
            
    args = ['python', command, dataset_dir]
    
    p = subprocess.Popen(args, stdout=subprocess.PIPE)
    
    return p.pid
    
    
    
if __name__ == '__main__':
    
    pipeline_control()
    