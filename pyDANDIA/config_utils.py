######################################################################
#
# config.py - returns the config.json file as a python dictionary.
#           - change the value of a given key
######################################################################

import json
import sys
from os import path

def read_config(path_to_config_file):

    if path.isfile(path_to_config_file) == False:
        raise IOError("No config file found at given location: "+path_to_config_file)

    config_file = open(path_to_config_file,'r')

    config_dict = json.load(config_file)

    config_file.close()

    for key, value in config_dict.items():
        if '[' in value and ']' in value and ',' in value:
            entries = value.replace('[','').replace(']','').split(',')
            l = []
            for e in entries:
                try:
                    l.append(float(e))
                except:
                    l.append(e)
            config_dict[key] = l

    return config_dict

def set_config_value(path_to_config_file, key_name, new_value):


    if path.isfile(path_to_config_file) == False:
        raise IOError("No config file found at given location: "+path_to_config_file)

    config_file = open(path_to_config_file,'r')

    config_dict = json.load(config_file)

    config_file.close()

    config_dict[key_name]["value"] = new_value

    config_file = open(path_to_config_file,'w')

    json.dump(config_dict, config_file, ensure_ascii=True, indent=4, sort_keys=True)

    config_file.close()

    return True

def build_config_from_json(config_file):

    config_dict = read_config(config_file)

    config = {}
    for key, value in config_dict.items():
        config[key] = config_dict[key]['value']

    return config

def load_event_model(file_path, log):

    event_model = build_config_from_json(file_path)

    # Handle those keuywords which have list entries which may have None
    # entries
    dict_keys = ['source_fluxes', 'source_flux_errors',
                         'blend_fluxes', 'blend_flux_errors']

    for key in dict_keys:
        dd = event_model[key]
        new_dd = {}
        for ddkey,ddvalue in dd.items():
            if 'none' in str(ddvalue).lower():
                new_dd[ddkey] = None
            else:
                new_dd[ddkey] = ddvalue
        event_model[key] = new_dd

    keys = ['pi_E_N','pi_E_E','logq','logs','dsdt','dalphadt']
    for key in keys:
        if 'none' in str(event_model[key]).lower():
            event_model[key] = None
        if 'none' in str(event_model['sig_'+key]).lower():
            event_model['sig_'+key] = None

    log.info('Loaded the parameters of the event model from '+file_path)

    return event_model
