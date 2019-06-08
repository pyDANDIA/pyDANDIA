# -*- coding: utf-8 -*-
"""
Created on Fri Jun  7 14:39:29 2019

@author: rstreet
"""
import pipeline_setup
from os import path

class DataCollection:
    """Class describing a collection of datasets for a single field pointing"""
    
    def __init__(self):
        self.primary_ref = None
        self.red_dir = None
        self.data_list = []
        
    def get_datasets_for_reduction(self,datasets_file,log=None):
        """Method to read a list of the datasets to be reduced.
        
        The datasets file should be an ASCII text file with the following structure:
        RED_DIR /path/to/top/level/reduction/directory
        DATASET Name-of-dataset-sub-directory
        DATASET Name-of-dataset-sub-directory
        DATASET Name-of-dataset-sub-directory
        DATASET Name-of-dataset-sub-directory
        PRIMARY_REF Name-of-dataset-sub-directory (to be used as primary reference)
        
        Example file contents:
        RED_DIR /Users/rstreet/ROMEREA/2018/full_frame_test
        DATASET ROME-FIELD-16_lsc-doma-1m0-05-fl15_gp
        DATASET ROME-FIELD-16_lsc-doma-1m0-05-fl15_rp
        PRIMARY_REF ROME-FIELD-16_lsc-doma-1m0-05-fl15_ip
        """
        
        if path.isfile(datasets_file) == True:
    
            if log!= None:
                log.info('Found a reduce_datasets instruction file')
    
            file_lines = open(datasets_file).readlines()
                        
            if log!= None:
                log.info('Going to reduce the following datasets:')
            
            for line in file_lines:
                
                if len(line.replace('\n','')) > 0:
                    
                    entries = line.replace('\n','').split()
                    
                    if 'RED_DIR' in entries[0]:
                        self.red_dir = entries[1]
                        
                    elif 'DATASET' in entries[0]:
                        self.data_list.append(entries[1])
                        
                    elif 'PRIMARY' in entries[0]:
                        self.data_list.append(entries[1])
                        self.primary_ref = entries[1]
                                        
                if log!= None:
                    log.info(self.data_list[-1])
                
        else:
            
            if log!= None:
                log.info('No instruction file found, halting.')

    def summary(self):
        
        output = 'Reduction location: '+repr(self.red_dir)+'\n'
        output += ' Datasets:\n'
        for d in self.data_list:
            output += repr(d)+'\n'
        output += 'Primary reference: '+repr(self.primary_ref)
        
        return output

def build_pipeline_setup(data):
    """Function to configure the pipeline setup object"""

    params = { 'base_dir': data.red_dir, 
               'log_dir': data.red_dir,
               'pipeline_config_dir': path.join(data.red_dir,'config'),
               }
               
    setup = pipeline_setup.pipeline_setup(params)

    return setup
        