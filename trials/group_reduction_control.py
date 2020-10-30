from os import getcwd, path, remove
from sys import argv, exit
from sys import path as systempath
import copy
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import subprocess
import datetime as datetime
import glob
from pyDANDIA import config_utils
from pyDANDIA import logs
from pyDANDIA import stage3_db_ingest
from pyDANDIA import stage6

def reduce_group():
    """Driver function for the pyDANDIA pipelined reduction of a
    group of datasets for the same field.

    A single dataset is defined as a set of images taken with a consistent
    instrument configuration and filter of a single sky pointing.

    Input parameters:
        dataset_red_dir   str   Full path to the reduction directory for this
                                dataset
    """

    reduction_version = 'group_reduction_control v0.1'

    config = get_args()

    red_log = logs.start_pipeline_log(config['log_dir'], 'group_reduction_control',
                                  version=reduction_version)

    pid_list = []
    for dir in config['red_dir_list']:

        if dir == config['primary_ref_dir']:
            data_status = 'primary-ref'
        else:
            data_status = 'non-ref'

        pid = trigger_parallel_auto_reduction(config,dir,phot_db_path,
                                                data_status)

        pid_list.append(pid)

        log.info(' -> Dataset '+path.basename(dir)+\
                ' reduction PID '+str(pid))

    # Wait for all reduction processes to complete before proceeding
    exit_codes = [p.wait() for p in pid_list]

    # Build a new photometric database if one does not already exist
    if path.isfile(config['phot_db_path']) == False:

        # First for primary ref dataset
        (status,report) = stage3_db_ingest.run_stage3_db_ingest(setup, primary_ref=True)

        # Then for the rest of the datasets:
        for dir in config['red_dir_list']:
            if dir != config['primary_ref_dir']:
                (status,report) = stage3_db_ingest.run_stage3_db_ingest(setup, primary_ref=False)

    # Run stage 6 for all datasets in sequence
    for dir in config['red_dir_list']:
        dparams = copy.copy(params)
        dparams['red_dir'] = dataset
        dparams['log_dir'] = path.join(dparams['red_dir'],'..','logs')
        dparams['base_dir'] = path.join(dparams['red_dir'],'..')

        dsetup = pipeline_setup.pipeline_setup(dparams)

        red_log.info('Ingesting '+path.basename(dataset))

        (status,report) = stage6.run_stage6(dsetup)

    logs.close_log(red_log)

def get_args():
    """Function to obtain the command line arguments necessary to run a
    single-dataset reduction."""

    helptext = """
                    pyDANDIA Data Group Reduction Control

    Main driver program to run pyDANDIA in pipeline mode for a data group
    consisting of multiple datasets of the same field.

    Command and options:
    > python group_reduction_control.py phot_db_path mode primary_dir red_dir1 red_dir2... [-v N ]

    where red_dir1, red_dir2... are the path to the reduction directories of datasets in this data group
          phot_db_path is the path to a photometry database
          mode is the mode of reduction required
          primary__dir indicates the path to the dataset which is to be used as a primary reference

    Reduction mode options are:
          mode  ['auto']
    """

    if '-help' in argv:
        print(helptext)
        exit()

    reduction_modes = ['auto']

    config = {}

    if len(argv) < 6:

        print(helptext)

    else:

        config['config_path'] = argv[1]
        config['db_file_path'] = argv[2]
        config['mode'] = argv[3]
        config['primary_dir'] = argv[4]
        config['red_dir_list'] = argv[5:]

    json_dict = config_utils.build_config_from_json(config['config_file'])

    for key, value in json_dict:
        config[key] = value

    return config

if __name__ == '__main__':
    reduce_group()
