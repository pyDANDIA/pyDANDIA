# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 17:04:53 2017

@author: rstreet
"""

from os import getcwd, path, remove
from sys import argv, exit
from sys import path as systempath
#cwd = getcwd()
#systempath.append(path.join(cwd,'../'))
from pyDANDIA import pipeline_setup
from pyDANDIA import stage0
from pyDANDIA import stage1
from pyDANDIA import stage2
from pyDANDIA import stage3
from pyDANDIA import stage4
from pyDANDIA import stage4b
from pyDANDIA import stage5
#from pyDANDIA.db import astropy_interface
#from pyDANDIA import stage6
from pyDANDIA import starfind
from pyDANDIA import logs

def run_stage_stand_alone():
    """Function to run a stage or section of pyDANDIA in stand alone mode."""
    
    params = get_args()
    
    setup = pipeline_setup.pipeline_setup(params)
    
    if params['stage'] == 'stage0':
        
        (status, report, reduction_metadata) = stage0.run_stage0(setup)
        
    elif params['stage'] == 'stage1':
        
        (status, report) = stage1.run_stage1(setup)

    elif params['stage'] == 'starfind':
        reduction_metadata = metadata.MetaData()
        try:
            reduction_metadata.load_all_metadata(metadata_directory=setup.red_dir,
                                                 metadata_name='pyDANDIA_metadata.fits')
        
        except:
            status = 'ERROR'
            report = 'Could not load the metadata file.'
            return status, report

        (status, report) = starfind.run_starfind(setup, reduction_metadata)

    elif params['stage'] == 'stage2':
        
        (status, report) = stage2.run_stage2(setup)

    elif params['stage'] == 'stage3':
        
        (status, report) = stage3.run_stage3(setup)

    elif params['stage'] == 'stage4':

        (status, report) = stage4.run_stage4(setup)

    elif params['stage'] == 'stage4b':
        
        (status, report) = stage4b.run_stage4b(setup)
        
    elif params['stage'] == 'stage5':
        
        (status, report) = stage5.run_stage5(setup)

    elif params['stage'] == 'stage6':
        
        (status, report) = stage6.run_stage6(setup)
    else:
        
        print('ERROR: Unsupported stage name given')
        exit()
    
    print('Completed '+params['stage']+' with status:')
    print(repr(status)+': '+str(report))
    

def get_args():
    """Function to acquire the commandline arguments needed to run a stage
    of pyDANDIA in stand alone mode."""
    
    helptext = """              RUN STAGE STAND ALONE
            
            Call sequence is:
            > python run_stage.py [stage] [path to reduction directory] [-v]
            
            where stage is the name of the stage or code to be run, one of:
                stage0, stage1, stage2, stage3, stage4, stage5, starfind
            
            and the path to the reduction directory is given to the dataset
            to be processed
            
            The optional -v flag controls the verbosity of the pipeline 
            logging output.  Values 
            N can be:
            -v 0 [Default] Essential logging output only, written to log file. 
            -v 1           Detailed logging output, written to log file.
            -v 2           Detailed logging output, written to screen and to log file.
            """
    
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
    
