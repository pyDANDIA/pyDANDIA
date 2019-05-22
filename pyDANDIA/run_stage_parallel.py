# -*- coding: utf-8 -*-
"""
Created on Thu Mar  7 14:40:52 2019

@author: rstreet
"""

from os import getcwd, path, remove
from sys import argv
import subprocess

from pyDANDIA import pipeline_setup
from pyDANDIA import pipeline_control
from pyDANDIA import run_stage
from pyDANDIA import stage0
from pyDANDIA import stage1
from pyDANDIA import stage2
from pyDANDIA import stage3
from pyDANDIA import stage3_db_ingest
from pyDANDIA import stage4
from pyDANDIA import stage6
from pyDANDIA import logs

def run_stage_parallax():
    """Function to run a stage or section of pyDANDIA on multiple datasets."""
    
    params = get_args()
    
    setup = pipeline_setup.pipeline_setup(params)
    
    log = logs.start_pipeline_log(setup.log_dir, 'run_stage_parallel')

    datasets = pipeline_control.get_datasets_for_reduction(setup,log)
    print(datasets)
    run_parallel_reductions(params,setup,datasets,log)
    
    logs.close_log(log)


def get_args():
    
    params = {}
    
    if len(argv) < 3:
        params['base_dir'] = input('Please enter the path to the base directory: ')
        params['stage'] = input('Please enter the name of the stage or code you wish to run: ')
        
    else:
        params['base_dir'] = argv[1]
        params['stage'] = argv[2]
        
    params['log_dir'] = path.join(params['base_dir'],'logs')
    params['pipeline_config_dir'] = path.join(params['base_dir'],'config')
    
    return params
    
def run_parallel_reductions(params,setup,datasets,log):
    
    command = path.join(setup.software_dir,'run_stage.py')
    
    log.info('Running '+params['stage']+' reductions:')
    
    for data_dir in datasets:
        
        dataset_dir = path.join(setup.base_dir,data_dir)
        
        args = ['python', command, params['stage'], dataset_dir]
        
        p = subprocess.Popen(args, stdout=subprocess.PIPE)
        
        log.info(' -> Dataset '+path.basename(dataset_dir)+\
                ' reduction PID '+str(p.pid))


if __name__ == '__main__':
    
    run_stage_parallax()