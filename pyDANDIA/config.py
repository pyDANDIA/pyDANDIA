######################################################################
#                                                                   
# config.py - returns the config.json file as a python dictionary.
#           - change the value of a given key
######################################################################

import json
import sys

def read_config(path_to_config_file):
    try:
        config_file = open(path_to_config_file,'r')
    except:
        print "No config file found at given location."
	sys.exit()
    try:
        config_dict = json.load(config_file)
    except:
        print "config.json file failed to load."
	sys.exit()
    config_file.close()
    return config_dict


def set_config_value(path_to_config_file, key_name, new_value):
    try:
        config_file = open(path_to_config_file,'r')
    except:
        print "No config file found at given location."
	sys.exit()
    try:
        config_dict = json.load(config_file)
    except:
        print "config.json file failed to load."
	sys.exit()
    config_file.close()
    config_dict[key_name]["value"] = new_value
    try:
        config_file = open(path_to_config_file,'w')
    except:
        print "Failed to open config.json for writing."
	sys.exit()
    try:
        json.dump(config_dict, config_file, ensure_ascii=True, indent=4, sort_keys=True)
    except:
        print "Failed to write to config.json."
    config_file.close()
    return True

