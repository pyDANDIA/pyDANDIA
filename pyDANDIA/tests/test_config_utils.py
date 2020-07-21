# -*- coding: utf-8 -*-
"""
Created on Tue Oct 10 08:50:19 2017

@author: rstreet
"""

from os import getcwd, path, remove
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
from shutil import copyfile
import config_utils
import logs

TEST_DATA = path.join(cwd,'data')

def test_read_config():
    """Function to unit test the reading of the pipeline's JSON config files"""

    config_file_path = '../../Config/config.json'

    config = config_utils.read_config(config_file_path)

    assert type(config) == type({'a':1, 'b':2})
    assert 'proc_data' in config.keys()

    config_file_path = '../../Config/inst_config.json'

    config = config_utils.read_config(config_file_path)

    assert type(config) == type({'a':1, 'b':2})
    assert 'instrid' in config.keys()


def test_set_config_value():

    test_key = 'proc_data'
    new_value = '/Users/test/path'

    config_file_path = path.join(TEST_DATA,'config.json')
    copyfile('../../Config/config.json',config_file_path)

    init_config = config_utils.read_config(config_file_path)

    status = config_utils.set_config_value(config_file_path, test_key, new_value)

    updated_config = config_utils.read_config(config_file_path)

    assert updated_config[test_key]['value'] == new_value

    remove(config_file_path)

def generate_test_event_model(file_path):

    f = open(file_path, 'w')
    f.write("""{
      "t0": {
            "value": 2458551.30979,
            "comment": "Time of event peak",
            "format": "float",
            "unit": "days [HJD]"
        },
        "sig_t0": {
            "value": 0.28307,
            "comment": "Uncertainty on time of event peak",
            "format": "float",
            "unit": "days [HJD]"
        },
        "tE": {
            "value": 2458551.30979,
            "comment": "Time of event peak",
            "format": "float",
            "unit": "days [HJD]"
        },
        "sig_tE": {
            "value": 0.28307,
            "comment": "Uncertainty on time of event peak",
            "format": "float",
            "unit": "days [HJD]"
        },
        "u0": {
            "value": -0.28594,
            "comment": "Impact parameter",
            "format": "float",
            "unit": ""
        },
        "sig_u0": {
            "value": 0.00402,
            "comment": "Uncertainty on impact parameter",
            "format": "float",
            "unit": ""
        },
        "rho": {
            "value": 0.00342,
            "comment": "Source angular size parameter",
            "format": "float",
            "unit": ""
        },
        "sig_rho": {
            "value": 0.0023,
            "comment": "Uncertainty on source angular size parameter",
            "format": "float",
            "unit": ""
        },
        "pi_E": {
            "value": 0.29290,
            "comment": "Parallax",
            "format": "float",
            "unit": ""
        },
        "sig_pi_E": {
            "value": 0.0181,
            "comment": "Uncertainty on parallax",
            "format": "float",
            "unit": ""
        },
        "logq": {
            "value": -1.37703,
            "comment": "Lens mass ratio, log10",
            "format": "float",
            "unit": ""
        },
        "sig_logq": {
            "value": 0.01129,
            "comment": "Uncertainty on lens mass ratio, log10",
            "format": "float",
            "unit": ""
        },
        "logs": {
            "value": 0.22445,
            "comment": "Lens masses separation, log10",
            "format": "float",
            "unit": ""
        },
        "sig_logs": {
            "value": 0.00153,
            "comment": "Uncertainty on lens mass separation, log10",
            "format": "float",
            "unit": ""
        },
        "dsdt": {
            "value": "None",
            "comment": "Rate of change in lens masses separation",
            "format": "float",
            "unit": ""
        },
        "sig_dsdt": {
            "value": "None",
            "comment": "Uncertainty on rate of change of lens mass separation",
            "format": "float",
            "unit": ""
        },
        "dalphadt": {
            "value": "None",
            "comment": "Rate of change in alpha",
            "format": "float",
            "unit": "radians/s"
        },
        "sig_dalphadt": {
            "value": "None",
            "comment": "Uncertainty on rate of change of alpha",
            "format": "float",
            "unit": "radians/s"
        },
        "source_fluxes": {
            "value": {"lsc-doma-fa15_gp": 28.72093,
                      "lsc-doma-fa15_rp": 1251.48254,
                      "lsc-doma-fa15_ip": 7784.05576},
            "comment": "Measured source flux values in SDSS g, r, and i bands, in order or list with None entries",
            "format": "float",
            "unit": "e-/s"
        },
        "source_flux_errors": {
            "value": {"lsc-doma-fa15_gp": 13.53316,
                      "lsc-doma-fa15_rp": 41.82985,
                      "lsc-doma-fa15_ip": 149.234},
            "comment": "Uncertainties on measured source flux values in SDSS g, r, and i bands, in order or list with None entries",
            "format": "float",
            "unit": "e-/s"
        },
        "blend_fluxes": {
            "value": {"lsc-doma-fa15_gp": 1723.43824,
                      "lsc-doma-fa15_rp": 8078.1425,
                      "lsc-doma-fa15_ip": 21112.69562},
            "comment": "Measured blend flux values in SDSS g, r, and i bands, in order or list with None entries",
            "format": "float",
            "unit": "e-/s"
        },
        "blend_flux_errors": {
            "value": {"lsc-doma-fa15_gp": 16.85739,
                      "lsc-doma-fa15_rp": 49.10606,
                      "lsc-doma-fa15_ip": 155.75005},
            "comment": "Uncertainties on measured blend flux values in SDSS g, r, and i bands, in order or list with None entries",
            "format": "float",
            "unit": "e-/s"
        }
    }""")
    f.close()

def test_load_event_model():

    log = logs.start_stage_log( cwd, 'test_config_utils' )

    file_path = 'data/test_event_model.json'

    generate_test_event_model(file_path)

    event_model = config_utils.load_event_model(file_path, log)
    print(event_model)

    assert type(event_model) == type({})

    keys = ['t0','tE','u0','rho','pi_E_E','pi_E_N','logq','logs','dsdt','dalphadt']
    for k in keys:
        assert k in event_model.keys()
        assert 'sig_'+k in event_model.keys()

    keys = ['source_fluxes', 'source_flux_errors',
                         'blend_fluxes', 'blend_flux_errors']
    for k in keys:
        assert type(event_model[k]) == type({})

    logs.close_log(log)

def test_load_analyse_cmd_config():

    config_file = '../../config/field_colour_analysis.json'

    config = config_utils.build_config_from_json(config_file)

    keys = ['flux_reference_datasets']
    for k in keys:
        assert type(config[k]) == type({})

def test_load_auto_config():

    config_file = '../../config/auto_pipeline_config.json'

    config = config_utils.build_config_from_json(config_file)

    for key, value in config.items():
        print(key+': '+repr(value))

if __name__ == '__main__':

    #test_load_event_model()
    #test_load_analyse_cmd_config()
    test_load_auto_config()
