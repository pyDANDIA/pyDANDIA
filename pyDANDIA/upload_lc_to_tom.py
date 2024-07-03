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

    if log:
        log.info('Started lightcurve upload with payload: ' + repr(payload))

    config_file = path.join(setup.pipeline_config_dir, 'tom_config.json')
    config = config_utils.build_config_from_json(config_file)
    login = (config['user_id'], config['password'])

    decision_to_upload = decide_whether_to_upload(payload, config, login, log=log)

    if decision_to_upload:

        reduction_metadata = metadata.MetaData()
        reduction_metadata.load_all_metadata(setup.red_dir, 'pyDANDIA_metadata.fits')
        payload['name'] = reduction_metadata.headers_summary[1]['OBJKEY'][0]

        close_log_file = False
        if log==None:
            log = logs.start_stage_log( setup.red_dir, 'mop_upload' )
            close_log_file = True

        print(config, login, payload)
        (target_pk, target_groups) = tom.get_target_id(config, login, payload, log=log)
        print(target_pk, target_groups)

        if target_pk:
            existing_datafiles = tom.list_dataproducts(config, login, payload, target_pk, log=log)

            tom.delete_old_datafile_version(config, login, payload, existing_datafiles, log=log)

        tom.upload_datafile(config, login, payload, target_pk, target_groups, log=log)

        if close_log_file:
            logs.close_log(log)

def decide_whether_to_upload(payload, config, login, log=None):
    upload = True

    if not path.isfile(payload['file_path']):
        upload = False
        if log!=None:
            log.info('-> Lightcurve upload to TOM aborted due to missing data file '+payload['file_path'])

    file_lines = open(payload['file_path'],'r').readlines()
    if len(file_lines) < 6:     # Allows for header line
        upload = False
        if log!=None:
            log.info('-> Lightcurve has too few datapoints ('+str(len(file_lines))+') to upload to TOM ')

    if upload and log!=None:
        log.info('-> Lightcurve has passed sanity checks and will be uploaded to TOM')

    mop_status = tom.check_mop_live(config, login)
    if log:
        log.info('-> MOP status: '+repr(mop_status))
    if not mop_status:
        upload = False

    return upload

def get_args():
    if len(argv) == 1:
        red_dir = input('Please enter the path to reduction directory: ')
        file_path = input('Please enter the path to the data file: ')
        suffix = input('Please enter the filename suffix, or press return for none: ')
    else:
        red_dir = argv[1]
        file_path = argv[2]
        suffix = argv[3]

    payload = {'file_path': file_path}
    search_string = path.basename(red_dir)
    if len(suffix) > 0:
        search_string = search_string+'_'+suffix
    payload = {'file_path': file_path, 'search_string': search_string}
    setup = pipeline_setup.pipeline_setup({'red_dir': red_dir})
    log = logs.start_pipeline_log(red_dir, 'tom_upload')
    print(setup.red_dir, setup.pipeline_config_dir)
    
    return setup, payload, log

if __name__ == '__main__':
    (setup, payload, log) = get_args()
    upload_lightcurve(setup, payload, log=log)
    logs.close_log(log)
