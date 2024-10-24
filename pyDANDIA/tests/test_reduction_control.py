# -*- coding: utf-8 -*-
"""
Created on Mon Aug 27 10:44:15 2018

@author: rstreet
"""


import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import logs
import pipeline_setup
import metadata
import reduction_control
import stage0
import glob
from astropy.table import Column

TEST_DATA = os.path.join(cwd,'data')

VERSION = 'test_reduction_control v0.1'

params = {'red_dir': os.path.join(cwd, 'data', 'proc',
                                   'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip'),
              'log_dir': os.path.join(cwd, 'data', 'proc',
                                   'logs'),
              'pipeline_config_dir': os.path.join(cwd, 'data', 'proc',
                                   'config'),
              'software_dir': os.path.join(cwd, '..'),
              'verbosity': 2
            }

def prepare_test_directory(setup):
    """Function to clean the test data directory of old data products"""

    stage_log = os.path.join(setup.red_dir,'stage0.log')
    metadata = os.path.join(setup.red_dir,'pyDANDIA_metadata.fits')

    if os.path.isfile(stage_log):
        os.remove(stage_log)

    return stage_log, metadata

def test_trigger_stage_subprocess():
    """Function to test the running of a stage as a subprocess"""

    setup = pipeline_setup.pipeline_setup(params)

    log = logs.start_pipeline_log(setup.log_dir, 'test_reduction_control',
                               version=VERSION)

    (stage_log,metadata) = prepare_test_directory(setup)

    process = reduction_control.trigger_stage_subprocess('stage0',setup,log,
                                                         wait=True)

    assert type(process.pid) == type(0)
    assert os.path.isfile(stage_log) == True
    assert os.path.isfile(metadata) == True

    logs.close_log(log)

def test_execute_stage():
    """Function to test the execution of a stage direct from a function"""

    setup = pipeline_setup.pipeline_setup(params)

    log = logs.start_pipeline_log(setup.log_dir, 'test_reduction_control',
                               version=VERSION)

    (stage_log,metadata) = prepare_test_directory(setup)

    status = 'OK'
    status = reduction_control.execute_stage(stage0.run_stage0, 'stage 0',
                                             setup, status, log)

    assert os.path.isfile(stage_log) == True
    assert os.path.isfile(metadata) == True

    logs.close_log(log)

def test_check_dataset_lock():

    setup = pipeline_setup.pipeline_setup(params)

    lockfile = os.path.join(setup.red_dir,'dataset.lock')

    if os.path.isfile(lockfile):
        os.remove(lockfile)

    log = logs.start_pipeline_log(setup.log_dir, 'test_reduction_control',
                               version=VERSION)

    status = reduction_control.check_dataset_lock(setup,log)

    assert status == False

    f = open(lockfile,'w')
    f.write('test')
    f.close()

    status = reduction_control.check_dataset_lock(setup,log)

    assert status == True

    os.remove(lockfile)

    logs.close_log(log)

def test_lock_dataset():

    setup = pipeline_setup.pipeline_setup(params)

    lockfile = os.path.join(setup.red_dir,'dataset.lock')

    if os.path.isfile(lockfile):
        os.remove(lockfile)

    log = logs.start_pipeline_log(setup.log_dir, 'test_reduction_control',
                               version=VERSION)

    reduction_control.lock_dataset(setup,log)

    assert os.path.isfile(lockfile)

    if os.path.isfile(lockfile):
        os.remove(lockfile)

    logs.close_log(log)

def test_unlock_dataset():

    setup = pipeline_setup.pipeline_setup(params)

    lockfile = os.path.join(setup.red_dir,'dataset.lock')

    f = open(lockfile,'w')
    f.write('test')
    f.close()

    log = logs.start_pipeline_log(setup.log_dir, 'test_reduction_control',
                               version=VERSION)

    reduction_control.unlock_dataset(setup,log)

    assert os.path.isfile(lockfile) == False

    if os.path.isfile(lockfile):
        os.remove(lockfile)

    logs.close_log(log)

def test_get_auto_config():

    setup = pipeline_setup.pipeline_setup(params)

    log = logs.start_pipeline_log(setup.log_dir, 'test_reduction_control',
                               version=VERSION)

    config = reduction_control.get_auto_config(setup,log)

    for key, value in config.items():
        print(key+': '+repr(value))

    logs.close_log(log)

def test_extract_target_lightcurve():

    if len(sys.argv) > 1:
        red_dir = sys.argv[1]
        phot_db_path = sys.argv[2]
    else:
        red_dir = input('Please give the path to a reduced data directory: ')
        phot_db_path = input('Please give the path to the photometry DB: ')

    test_params = {'red_dir': red_dir,
              'log_dir': os.path.join(cwd, 'data', 'proc','logs'),
              'db_file_path': phot_db_path,
              'pipeline_config_dir': os.path.join(cwd, 'data', 'proc', 'config'),
              'software_dir': os.path.join(cwd, '..'),
              'verbosity': 2,
              'project_id': 'TEST_PROJECT',
              'phot_error_threshold': 0.05}

    lc_dir = os.path.join(red_dir, 'lc')

    test_setup = pipeline_setup.pipeline_setup(test_params)

    log = logs.start_pipeline_log(test_setup.log_dir, 'test_reduction_control',
                               version=VERSION)

    reduction_control.extract_target_lightcurve(test_setup, test_params, log)

    logs.close_log(log)

    lc_files = glob.glob(os.path.join(lc_dir, '*'))

    assert len(lc_files) > 0

def test_check_for_assigned_ref_image():

    setup = pipeline_setup.pipeline_setup(params)

    log = logs.start_pipeline_log(setup.log_dir, 'test_reduction_control',
                                   version=VERSION)

    metadata_path = os.path.join(setup.red_dir, 'pyDANDIA_metadata.fits')
    if os.path.isfile(metadata_path):
        os.remove(metadata_path)

    reduction_metadata = metadata.MetaData()
    reduction_metadata.create_metadata_file(setup.red_dir, 'pyDANDIA_metadata.fits')
    reduction_metadata.save_updated_metadata(setup.red_dir,'pyDANDIA_metadata.fits',log=log)

    status = reduction_control.check_for_assigned_ref_image(setup, log)

    assert status==False

    ref_path = os.path.join(setup.red_dir, 'ref')
    col1 = Column([str(ref_path)], name='REF_PATH')
    reduction_metadata.data_architecture[1].add_column(col1)

    col2 = Column(['test_ref_image.fits'], name='REF_IMAGE')
    reduction_metadata.data_architecture[1].add_column(col2)
    reduction_metadata.save_updated_metadata(setup.red_dir,'pyDANDIA_metadata.fits',log=log)

    status = reduction_control.check_for_assigned_ref_image(setup, log)

    assert status==True

    logs.close_log(log)

def test_get_lightcurve_attribution():
    test_project = 'OMEGA'
    test_params = [{'project_id': test_project, 'instrument': 'fa06'},
                    {'project_id': test_project, 'instrument': 'ep03'},
                    {'project_id': test_project, 'instrument': 'fl16'},
                    {'instrument': 'fa06'}]
    filter_name = 'ip'
    test_attributions = [test_project+'_sinistro_'+filter_name,
                        test_project+'_ep03_'+filter_name,
                        test_project+'_sinistro_'+filter_name,
                        'sinistro_'+filter_name]
    for i, config in enumerate(test_params):
        attribution = reduction_control.get_lightcurve_attribution(config, filter_name)
        assert attribution == test_attributions[i]

if __name__ == '__main__':

    #test_trigger_stage_subprocess()
    #test_execute_stage()
    #test_check_dataset_lock()
    #test_lock_dataset()
    #test_unlock_dataset()
    #test_get_auto_config()
    #test_extract_target_lightcurve()
    #test_check_for_assigned_ref_image()
    test_get_lightcurve_attribution()
