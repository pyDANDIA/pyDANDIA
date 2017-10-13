# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:04:53 2017

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
import starfind
import logs

def run_stage_stand_alone():
    """Function to run a stage or section of pyDANDIA in stand alone mode."""
    
    params = get_args()
    
    setup = pipeline_setup.pipeline_setup(params)
    
    if params['stage'] == 'stage0':
        
        (status, report, reduction_metadata) = stage0.run_stage0(setup)
        
    elif params['stage'] == 'stage1':
        
        (status, report) = stage1.run_stage1(setup)

    elif params['stage'] == 'starfind':
        
        (status, report) = starfind.run_starfind(setup)

    else:
        
        print('ERROR: Unsupported stage name given')
        exit()
    
    print('Completed '+params['stage']+' with status:')
    print(repr(status)+': '+str(report))
    

def get_args():
    """Function to acquire the commandline arguments needed to run a stage
    of pyDANDIA in stand alone mode."""
    
    helptext = '''              RUN STAGE STAND ALONE
            
            Call sequence is:
            > python run_stage.py [stage] [path to reduction directory]
            
            where stage is the name of the stage or code to be run, one of:
                stage0, stage1, starfind
            
            and the path to the reduction directory is given to the dataset
            to be processed
            '''
    
    params = {}
    
    if len(argv) == 1 or '-help' in argv:
        
        print(helptext)
        exit()
        
    if len(argv) < 3:
        
        params['stage'] = raw_input('Please enter the name of the stage or code you wish to run: ')
        params['red_dir'] = raw_input('Please enter the path to the reduction directory: ')
    
    else:
        
        params['stage'] = argv[1]
        params['red_dir'] = argv[2]    
    
    if '-v' in argv:
        
        idx = argv.index('-v')
        
        if len(argv) >= idx + 1:
            
            params['verbosity'] = int(argv[idx+1])
    
    return params

    
if __name__ == '__main__':
    
    run_stage_stand_alone()
    