# -*- coding: utf-8 -*-
"""
Created on Thu Jul 13 17:59:16 2017

@author: rstreet
"""
from os import path
from sys import exit, argv
import xml.sax
import metadata

def convert_red_metadata():
    """Function to convert the configuration parameter files and trendlogs
    from an (IDL) DanDIA reduction into the metadata structure used by
    pyDANDIA."""
    
    config = get_config()

    meta = metadata.MetaData()
    
    event_info = read_event_info(config)
    meta.set_pars(event_info)
    
    meta.set_reduction_paths(config['red_dir'])
    
    red_config = read_red_config(config)
    for key, value in red_config.items():
        meta.set_pars(red_config)
    
    meta.inventory = read_data_inventory(config)
    
    meta.imred = read_imred_trendlog(config['imred'])
    
    meta.gimred = read_gimred_trendlog(config['gimred'])
    
    meta.write()
        
def get_config():
    """Function to harvest user-specified configuration parameters"""
    
    config = {}
    
    if len(argv) > 1:
        config['red_dir'] = argv[1]
    else:
        config['red_dir'] = raw_input('Please enter the path to the DanDIA-format reduction directory: ')
    
    config['event_code'] = path.basename(config['red_dir'])
    config['filter'] = config['event_code'].split('_')[-1]
    config['event_info'] = path.join(config['red_dir'],\
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

    keys = {'Object:':'field' ,
            'Year:': 'year',
            'Network:': 'network',
            'Site:': 'site',
            'Enclosure:': 'enclosure',
            'Telescope:':'telescope',
            'Instrument:':'instrument',
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
    
    event_info['filter'] = dominant_filter(event_info)
    
    return event_info

def dominant_filter(event_info):
    """Function to decide which of the parsed filters is the dominant one."""
    
    flist = [str(event_info['f1']).lower(), 
                        str(event_info['f2']).lower(), 
                        str(event_info['f3']).lower()]
    filters = []
    for f in flist:
        if f != 'air' and f != 'none' and f != 'clean':
            filters.append(f)
    
    if len(filters) == 1:
        filter_name = filters[0]
    
    else:
        filter_name = '/'.join(filters)
        
    return filter_name
 
class ConfigHandler(xml.sax.handler.ContentHandler):
    """Parse config file parameters"""
    
    def __init__(self):
        self.iValue = 0
        self.mapping = {}
    
    def startElement(self,name,attributes):
        if name == 'parameter':
            self.buffer = ""
            self.paramName = attributes["name"]
        elif name == 'value':
            self.iValue = 1
    
    def characters(self, data):
        if self.iValue == 1:
            self.buffer += data
    
    def endElement(self, name):
        if name == 'value':
            self.iValue = 0
            self.mapping[self.paramName] = self.buffer
 

def read_red_config(config):
    """Function to read the parameters from an old-format Red.Config file"""
    
    red_config = {}
    if path.isfile(config['red_config']) == False:
        print('Error: Cannot find reduction configuration at '+config['red_config'])

    parser = xml.sax.make_parser(  )
    red_config = ConfigHandler( )
    parser.setContentHandler(red_config)
    parser.parse(config['red_config'])
    
    return red_config.mapping
    
def read_data_inventory(config):
    """Function to read the data inventory from an old-format Inventory file.
    Receives the full path to the file and returns a dictionary of the entries 
    of the form:
    Images[imname] = [eventname, eventname, imdate, imtime, procstatus, nproc]
    The framestatus parameter indicates what processing status of frame to 
    search for, ie whether to list (P)rocessed or (U)nprocessed frames.
    """
    
    inventory = []
    
    if path.isfile(config['inventory']) == False:
        print('Error: Cannot find the data inventory at '+config['inventory'])
        exit()
    
    file_lines = open(config['inventory'],'r').readlines()
    
    for line in file_lines:
        entries = line.split()
        inventory.append(entries)
    
    return inventory

def read_imred_trendlog(trend_file):
    """Function to read a generic-format trendlog from DanDIA in the old format"""
    
    header = ''
    data = {}

    if path.isfile(trend_file) == False:
        print('Error: Cannot find trendlog file '+trend_file)
        exit()
    
    file_lines = open(trend_file,'r').readlines()

    for line in file_lines:
        if len(line.replace('\n','')) > 0:
            if line[0:1] == '#':
                header = line.replace('\n','').split()
            else:
                entries = line.replace('\n','').split()
                tlist = [   float(entries[2]),\
                            float(entries[9]),\
                            float(entries[10]),\
                            float(entries[11]),\
                            float(entries[12]),\
                            int(entries[17]),\
                        ]
                data[path.basename(entries[0])] = tlist

    return data

def read_gimred_trendlog(trend_file):
    """Function to read a generic-format trendlog from DanDIA in the old format"""
    
    header = ''
    data = {}

    if path.isfile(trend_file) == False:
        print('Error: Cannot find trendlog file '+trend_file)
        exit()
    
    file_lines = open(trend_file,'r').readlines()

    for line in file_lines:
        if len(line) > 0:
            if line[0:1] == '#':
                header = line.replace('\n','').split()
            else:
                entries = line.replace('\n','').split()
                tlist = [   path.basename(entries[0]),
                            float(entries[3]),\
                            float(entries[4]),\
                            float(entries[5]),\
                            float(entries[6]),\
                            float(entries[7]),\
                            float(entries[8]),\
                            int(entries[10]),\
                            float(entries[11]),\
                            float(entries[12])
                        ]
                data[path.basename(entries[0])] = tlist

    return data


if __name__ == '__main__':
    convert_red_metadata()
    