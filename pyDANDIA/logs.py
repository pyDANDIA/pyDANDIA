# -*- coding: utf-8 -*-
"""
Created on Thu Jul 20 12:43:20 2017

@author: rstreet
"""

import logging
from os import path, remove
from sys import exit
from astropy.time import Time
from datetime import datetime
import glob

def start_stage_log( log_dir, stage_name, version=None ):
    """Function to initialize a log file for a single stage of pyDANDIA.  
    
    The naming convention for the file is [stage_name].log.  
    
    The new file will automatically overwrite any previously-existing logfile
    for the given reduction.  

    This function also configures the log file to provide timestamps for 
    all entries.  
    
    Parameters:
        log_dir   string        Directory path
                                log_root_name  Name of the log file
        stage_name  string      Name of the stage to be logged
                                Used as both the log file name and the name
                                of the logger Object 
        version   string        [optional] Stage code version string
    Returns:
        log       open logger object
    """
    
    # Console output not captured, though code remains for testing purposes
    console = False

    log_file = path.join(log_dir, stage_name+'.log')
    if path.isfile(log_file) == True:
        remove(log_file)
        
    # To capture the logging stream from the whole script, create
    # a log instance together with a console handler.  
    # Set formatting as appropriate.
    log = logging.getLogger( stage_name )
    
    if len(log.handlers) == 0:
        log.setLevel( logging.INFO )
        file_handler = logging.FileHandler( log_file )
        file_handler.setLevel( logging.INFO )
        
        if console == True:
            console_handler = logging.StreamHandler()
            console_handler.setLevel( logging.INFO )
    
        formatter = logging.Formatter( fmt='%(asctime)s %(message)s', \
                                    datefmt='%Y-%m-%dT%H:%M:%S' )
        file_handler.setFormatter( formatter )

        if console == True:        
            console_handler.setFormatter( formatter )
    
        log.addHandler( file_handler )
        if console == True:            
            log.addHandler( console_handler )
    
    log.info( 'Started run of '+stage_name+'\n')
    if version != None:
        log.info('  Software version: '+version+'\n')
        
    return log

def ifverbose(log,setup,string):
    """Function to write to a logfile only if the verbose parameter in the 
    metadata is set to True"""
    
    if log != None and setup.verbosity >= 1:

        log.info(string)
        
    if setup.verbosity == 2:

        print(string)

def close_log(log):
    """Function to cleanly shutdown logging functions with a final timestamped
    entry.
    Parameters:
        log     logger Object
    Returns:
        None
    """
    
    log.info( 'Processing complete\n' )
    logging.shutdown()
    
def start_pipeline_log( log_dir, log_name, version=None ):
    """Function to initialize a log file for the pyDANDIA pipeline.  
    
    The naming convention for the file is [log_name]_<date_string>.log.  
    
    The new file will automatically overwrite any previously-existing logfile
    for the given reduction.  

    This function also configures the log file to provide timestamps for 
    all entries.  
    
    Parameters:
        log_dir   string        Directory path
                                log_root_name  Name of the log file
        log_name  string        Name of the stage to be logged
                                Used as both the log file name and the name
                                of the logger Object 
        version   string        [optional] Stage code version string
    Returns:
        log       open logger object
    """
    
    # Console output not captured, though code remains for testing purposes
    console = False

    ts = datetime.utcnow()

    log_file = path.join(log_dir, log_name+'_'+ts.strftime("%Y-%m-%d")+'.log')
        
    # To capture the logging stream from the whole script, create
    # a log instance together with a console handler.  
    # Set formatting as appropriate.
    log = logging.getLogger( log_name )
    
    if len(log.handlers) == 0:
        log.setLevel( logging.INFO )
        file_handler = logging.FileHandler( log_file )
        file_handler.setLevel( logging.INFO )
        
        if console == True:
            console_handler = logging.StreamHandler()
            console_handler.setLevel( logging.INFO )
    
        formatter = logging.Formatter( fmt='%(asctime)s %(message)s', \
                                    datefmt='%Y-%m-%dT%H:%M:%S' )
        file_handler.setFormatter( formatter )

        if console == True:        
            console_handler.setFormatter( formatter )
    
        log.addHandler( file_handler )
        if console == True:            
            log.addHandler( console_handler )
    
    log.info( 'Started pipeline run \n')
    if version != None:
        log.info('  Software version: '+version+'\n')
        
    return log
