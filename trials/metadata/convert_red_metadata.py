# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:59:16 2017

@author: rstreet
"""
from os import path
from sys import exit

def convert_red_metadata():
    """Function to convert the configuration parameter files and trendlogs
    from an (IDL) DanDIA reduction into the metadata structure used by
    pyDANDIA."""
    
    config = get_config()

    event_info = read_event_info(config)
    

def get_config():
    """Function to harvest user-specified configuration parameters"""
    
    config = {}
    
    if len(argv) > 1:
        config['red_dir'] = argv[1]
    else:
        config['red_dir'] = raw_input('Please enter the path to the DanDIA-format reduction directory: ')
    
    config['event_code'] = path.basename(config['red_dir'])
    config['filter'] = config['event_code'].split('_')[-1]
    config['event_file'] = path.join(config['red_dir'],\
                                config['event_code']+'.Event.Info')
    config['red_config'] = path.join(config['red_dir'],\
                                config['event_code']+'.Red.Config')
    config['inventory'] = path.join(config['red_dir'],\
                                config['event_code']+'.Data.Inventory')
    config['imred'] = path.join(config['red_dir'],'trends',\
                                'trendlog.imred.'+config['filter']+'.txt')
    config['gimred'] = path.join(config['red_dir'],'trends',\
                                'trendlog.gimred.'+config['filter']+'.txt')
    config['dimred'] = path.join(config['red_dir'],'trends',\
                                'trendlog.dimred.'+config['filter']+'.txt')
    
    return config
    
def read_event_info(config):
    """Function to read the parameters from an old-format Event.Info file"""

    keys = {'Object:':'object' ,
            'Year:': 'year',
            'Network:': 'network',
            'Site:': 'site',
            'Enclosure:': 'enclosure',
            'Telescope:':'tel',
            'Instrument:':'instrum',
            'Filter 1:':'f1',
            'Filter 2:':'f2',
            'Filter 3:':'f3',
            'BinX:':'binx',
            'BinY:':'biny'}
    event_info = {}
    
    if path.isfile(config['event_info']) == False:
        print('Error: Cannot find event info file at '+config['event_info'])
        exit()
    
    file_lines = open(config['event_info'],'r').readlines()
    
    for header in keys.keys():
        for line in file_lines:
            if line.__contains__(header) == True:
                event_info[keys[header]] = line.split()[-1]
                
    return event_info
      
    


if __name__ == '__main__':
    convert_red_metadata()
    