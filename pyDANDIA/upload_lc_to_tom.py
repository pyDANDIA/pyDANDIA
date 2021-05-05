from os import path
from sys import argv
import requests
from datetime import datetime
import json
from pyDANDIA import config_utils
from pyDANDIA import tom
from pyDANDIA import logs
from pyDANDIA import pipeline_setup
from pyDANDIA import metadata

def upload_lightcurve(setup, payload, log=None):

    config_file = path.join(setup.pipeline_config_dir, 'tom_config.json')
    config = config_utils.build_config_from_json(config_file)
    login = (config['user_id'], config['password'])

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')
    payload['name'] = reduction_metadata.headers_summary[1]['OBJKEY'][0]

    close_log_file = False
    if log==None:
        log = logs.start_stage_log( setup.red_dir, 'mop_upload' )
        close_log_file = True

    (target_pk, target_groups) = tom.get_target_id(config, login, payload, log=log)

    if target_pk:
        existing_datafiles = tom.list_dataproducts(config, login, payload, target_pk, log=log)

        tom.delete_old_datafile_version(config, login, payload, existing_datafiles, log=log)

    tom.upload_datafile(config, login, payload, target_pk, target_groups, log=log)

    if close_log_file:
        logs.close_log(log)

def get_args():
    if len(argv) == 1:
        red_dir = input('Please enter the path to reduction directory: ')
        file_path = input('Please enter the path to the data file: ')
    else:
        red_dir = argv[1]
        file_path = argv[2]

    payload = {'file_path': file_path}
    setup = pipeline_setup.pipeline_setup({'red_dir': red_dir})

    return setup, payload

if __name__ == '__main__':
    (setup, payload) = get_args()
    upload_lightcurve(setup, payload)
