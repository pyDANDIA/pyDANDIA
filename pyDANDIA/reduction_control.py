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
import reference_astrometry
import stage3
import stage4
import stage5
import stage6
import logs
import subprocess

def reduction_control():
    """Main driver function for the pyDANDIA pipelined reduction of an 
    individual dataset.  
    
    A single dataset is defined as a set of images taken with a consistent
    instrument configuration and filter of a single sky pointing.  
    
    Input parameters:
        dataset_red_dir   str   Full path to the reduction directory for this
                                dataset
    """

    reduction_version = 'reduction_control v0.3'

    setup = get_args()
    
    log = logs.start_pipeline_log(setup.red_dir, 'reduction_control',
                                  version=reduction_version)
    
    log.info('Pipeline setup: '+setup.summary()+'\n')
    
    if setup.red_mode == 'data_preparation':
        
        run_data_preparation(setup,log,select_ref=True)
        
    elif setup.red_mode == 'added_data_preparation':
        
        run_data_preparation(setup,log,select_ref=False)
        
    elif setup.red_mode == 'reference_analysis':
        
        run_reference_image_analysis(setup,log)
        
    elif setup.red_mode == 'image_analysis':
        
        run_image_analysis(setup,log)
        
    else:
        log.info('ERROR: unrecognised reduction mode ('+setup.red_mode+') selected')
        
    logs.close_log(log)

def run_data_preparation(setup,log=None,select_ref=False):
    """Function to run in sequence stages 0 - 2 for a single dataset"""

    if log!=None:
        log.info('Pipeline setup: '+setup.summary()+'\n')
    
    (status,report,meta_data) = stage0.run_stage0(setup)
    
    if log!=None:
        log.info('Completed stage 0 with status '+repr(status)+': '+report)
    
    status = execute_stage(stage1.run_stage1, 'stage 1', setup, status, log)
    
    if select_ref:
        status = execute_stage(stage2.run_stage2, 'stage 2', setup, status, log)
    
def run_reference_image_analysis(setup,log):
    """Function to run the pipeline stages which perform the analysis of a
    reference image in sequence."""
    
    log.info('Pipeline setup: '+setup.summary()+'\n')
    
    status = 'OK'
    
    status = execute_stage(reference_astrometry.run_reference_astrometry, 
                           'reference astrometry', setup, status, log)
    
    status = execute_stage(stage3.run_stage3, 'stage 3', setup, status, log)

def run_image_analysis(setup,log):
    """Function to run the sequence of stages which perform the image 
    subtraction and photometry for a dataset"""
    
    log.info('Pipeline setup: '+setup.summary()+'\n')
    
    status = 'OK'
    
    status = execute_stage(stage4.run_stage4, 'stage 4', setup, status, log)
    
    status = execute_stage(stage5.run_stage5, 'stage 5', setup, status, log)

def execute_stage(run_stage_func, stage_name, setup, status, log):
    """Function to execute a stage and verify whether it completed successfully
    before continuing.
    
    Accepts as an argument the status of the previous stage in order to check
    whether or not to continue the reduction.  If the reduction proceeds, this
    status is overwritten with the status output of the next stage. 
    
    Inputs:
        :param function run_stage_func: Single imported function
        :param string stage_name: Function name, for logging output
        :param object setup: pipeline setup object instance
        :param string status: Status of execution of the most recent stage
        :param logging log: open log file object
    
    Outputs:
        :param string status: Status of execution of the most recent stage
    """
    
    if 'OK' in status:
        
        if '0' in stage_name:
            
            (status, report, metadata) = run_stage_func(setup)
            
        else:
            
            (status, report) = run_stage_func(setup)
            
        log.info('Completed '+stage_name+' with status '+\
                    repr(status)+': '+report)
        
    if 'OK' not in status:
        
        log.info('ERROR halting reduction due to previous errors')
        
        logs.close_log(log)
        
        exit()
        
    return status

def parallelize_stages345(setup, status, log):
    """Function to execute stages 4 & 5 in parallel with stage 3.
    
    Inputs:
        :param object setup: pipeline setup object instance
        :param string status: Status of execution of the most recent stage
        :param logging log: open log file object
    
    Outputs:    
        :param string status: Status of execution of the most recent stage
    """
    
    log.info('Executing stage 3 in parallel with stages 4 & 5')
    
    process3 = trigger_stage_subprocess('stage3',setup,log,wait=False)
    
    process4 = trigger_stage_subprocess('stage4',setup,log,wait=True)
    process5 = trigger_stage_subprocess('stage5',setup,log,wait=True)
    
    log.info('Completed stages 4 and 5; now waiting for stage 3')
    
    (outs, errs) = process3.communicate()

    if errs == None:
        
        process3.wait()
        log.info('Completed stage 3')

    else:

        log.info('ERROR: Problem encountered in stage 3:')
        log.info(errs)
        
    log.info('Completed parallel stages')
    
    return 'OK'
    
def trigger_stage_subprocess(stage_code,setup,log,wait=True):
    """Function to run a stage as a separate subprocess
    
    Inputs:
        :param string stage_code: Stage to be run without spaces, e.g. stage0
        :param object setup: Pipeline setup instance
    """
    
    command = path.join(setup.software_dir,'run_stage.py')
    
    args = ['python', command, stage_code, setup.red_dir]
    
    p = subprocess.Popen(args, stdout=subprocess.PIPE)
    
    log.info('Started '+stage_code+', PID='+str(p.pid))
    
    if wait:
        
        log.info('Waiting for '+stage_code+' to finish')
        
        p.wait()
    
        log.info('Completed '+stage_code)
        
    return p
    
def get_args():
    """Function to obtain the command line arguments necessary to run a 
    single-dataset reduction."""
    
    helptext = """
                    pyDANDIA Reduction Control
    
    Main driver program to run pyDANDIA in pipeline mode for a single dataset. 
    
    Command and options:
    > python reduction_control.py red_dir_path mode [-v N ]
    
    where red_dir_path is the path to a dataset's reduction directory
          mode is the mode of reduction required
    
    Reduction mode options are:
          mode  new_reference
          
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
    
    reduction_modes = ['data_preparation',
                       'added_data_preparation',
                       'reference_analysis', 
                       'image_analysis']
    
    params = {}
    
    if len(argv) == 1:
        
        params['red_dir'] = input('Please enter the path to the datasets reduction directory: ')
        params['field'] = input('Please enter the field name: ')
        params['mode'] = input('Please enter the reduction mode, one of {'+','.join(reduction_modes)+'}: ')
    
    else:
        
        params['red_dir'] = argv[1]
        params['field'] = argv[2]
        params['mode'] = argv[3]
    
    if '-v' in argv:
        
        idx = argv.index('-v')
        
        if len(argv) >= idx + 1:
            
            params['verbosity'] = int(argv[idx+1])
        
    params['log_dir'] = path.join(params['red_dir'],'..','logs')
    params['pipeline_config_dir'] = path.join(params['red_dir'],'..','config')
    params['base_dir'] = path.join(params['red_dir'],'..')
    params['software_dir'] = getcwd()
    params['phot_db_path'] = path.join(params['red_dir'],'..',params['field']+'_phot.db')
    
    setup = pipeline_setup.pipeline_setup(params)
    setup.red_mode = params['mode']
    
    return setup
    
    
    
if __name__ == '__main__':
    reduction_control()
