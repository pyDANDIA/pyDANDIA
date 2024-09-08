# -*- coding: utf-8 -*-
"""
@author: rstreet
"""
from os import getcwd, path, remove
from pyDANDIA import automatic_pipeline
from pyDANDIA import logs
from pyDANDIA import config_utils
import psutil
from datetime import datetime, timedelta

cwd = getcwd()
TEST_DATA = path.join(cwd,'data')
TEST_DIR = path.join(cwd,'data','proc',
                        'ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')
TEST_CONFIG_FILE = path.join(cwd,'../../config/auto_pipeline_config.json')

def test_config():

    config = {
        'data_red_dir': path.join(cwd,'data','proc'),
        'log_dir': path.join(cwd,'data','proc', 'logs'),
        'config_dir': path.join(cwd,'../../config/'),
        'phot_db_dir': path.join(cwd,'data','proc'),
        'software_dir': path.join(cwd,'../'),
        'group_processing_limit': 5,
        'catalog_xmatch': False,
        'use_gaia_phot': True,
        'reduce_datasets': 'ALL'
    }

    return config

def test_parse_configured_datasets():

    log = logs.start_stage_log( cwd, 'test_auto' )

    config = test_config()

    config['reduce_datasets'] = 'ALL'
    datasets = automatic_pipeline.parse_configured_datasets(config,log)

    assert type(datasets) == type(['foo'])
    assert len(datasets) >= 1
    assert TEST_DIR in datasets

    config['reduce_datasets'] = '@dataset_list.txt'
    f = open('dataset_list.txt','w')
    f.write('ROME-FIELD-0002_lsc-doma-1m0-05-fl15_ip')
    f.close()

    datasets = automatic_pipeline.parse_configured_datasets(config,log)

    assert type(datasets) == type(['foo'])
    assert len(datasets) >= 1
    assert TEST_DIR in datasets

    remove('dataset_list.txt')

    logs.close_log(log)

def test_check_dataset_dir_structure():

    log = logs.start_stage_log( cwd, 'test_auto' )

    status = automatic_pipeline.check_dataset_dir_structure(TEST_DIR, log)
    assert status == True

    status = automatic_pipeline.check_dataset_dir_structure(TEST_DATA, log)
    assert status == False

    logs.close_log(log)

def test_check_dataset_dir_lock():

    log = logs.start_stage_log( cwd, 'test_auto' )

    lockfile = path.join(TEST_DIR, 'dataset.lock')
    if path.isfile(lockfile):
        remove(lockfile)

    status = automatic_pipeline.check_dataset_dir_unlocked(TEST_DIR, log)
    assert status == True

    f = open(lockfile,'w')
    f.close()
    status = automatic_pipeline.check_dataset_dir_unlocked(TEST_DIR, log)
    assert status == False
    remove(lockfile)

    logs.close_log(log)

def test_sanity_check_data_before_reduction():

    log = logs.start_stage_log( cwd, 'test_auto' )

    datasets = [ TEST_DIR ]
    sane_datasets = automatic_pipeline.sanity_check_data_before_reduction(datasets,log)

    assert datasets == sane_datasets

    logs.close_log(log)

def test_check_process_status():

    log = logs.start_stage_log( cwd, 'test_auto' )

    pid_list = psutil.pids()
    pids = {}
    for p in pid_list[0:10]:
        pids[str(p)] = p

    running_processes = automatic_pipeline.check_process_status(pids,log)

    assert type(running_processes) == type({})
    assert len(running_processes) == len(pids)

    logs.close_log(log)

def test_check_dataset_for_recent_data():

    log = logs.start_stage_log( cwd, 'test_auto' )
    dt = timedelta(days=30.0)

    date_threshold = datetime.strptime("2017-06-30","%Y-%m-%d") - dt
    recent_data = automatic_pipeline.check_dataset_for_recent_data(date_threshold,TEST_DIR,log)
    assert(recent_data == True)

    date_threshold = datetime.utcnow() - dt
    recent_data = automatic_pipeline.check_dataset_for_recent_data(date_threshold,TEST_DIR,log)
    assert(recent_data == False)

    logs.close_log(log)

def test_identify_recent_data():

    log = logs.start_stage_log( cwd, 'test_auto' )

    config = {'data_red_dir': TEST_DIR, 'lookback_time': 30.0}
    datasets = automatic_pipeline.identify_recent_data(config, log)

    assert len(datasets) == 0
    assert type(datasets) == type(['foo'])

    logs.close_log(log)

def test_check_dataset_spectra():
    dir_list = {
        '/data/messier/omega/data/data_reduction/ASASSN-24fs_gp': False,
        '/data/messier/omega/data/data_reduction/ASASSN-24fs_en06_air': True,
    }

    for dir_path, test_status in dir_list.items():
        status = automatic_pipeline.check_dataset_spectra(dataset_path)
        assert(status == test_status)