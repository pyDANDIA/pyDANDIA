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
        print("No config file found at given location: "+path_to_config_file)
        sys.exit()

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
        print("No config file found at given location: "+path_to_config_file)
        sys.exit()

    config_file = open(path_to_config_file,'r')
    
    config_dict = json.load(config_file)
    
    config_file.close()
    
    config_dict[key_name]["value"] = new_value
    
    config_file = open(path_to_config_file,'w')
    
    json.dump(config_dict, config_file, ensure_ascii=True, indent=4, sort_keys=True)
    
    config_file.close()

    return True
