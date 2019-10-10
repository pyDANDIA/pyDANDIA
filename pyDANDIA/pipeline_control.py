# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:09:55 2017

@author: rstreet
"""
from os import getcwd, path, remove
from sys import argv
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import pipeline_setup
import glob
import subprocess
import logs

def pipeline_control():
    """Main driver program controlling the reduction of multiple datasets
    with pyDANDIA.
    """

    pipeline_version = 'pipeline_control v0.2'

    setup = get_args()

    log = logs.start_pipeline_log(setup.log_dir, 'pipeline_control',
                               version=pipeline_version)

    datasets = get_datasets_for_reduction(setup,log)

    run_reductions(setup,log,datasets)

    logs.close_log(log)


def get_args():
    """Function to acquire the necessary commandline arguments to run
    pyDANDIA in pipeline mode."""

    params = {}

    if len(argv) != 4:
        params['base_dir'] = input('Please enter the path to the base directory: ')
        params['phot_db_path'] = input('Please enter the path to the database file: ')
        print('''Please enter the required reduction mode out of:
        {data_preparation, added_data_preparation, reference_analysis, image_analysis}''')
        params['red_mode'] = input('Reduction mode: ')

    else:
        params['base_dir'] = argv[1]
        params['phot_db_path'] = argv[2]
        params['red_mode'] = argv[3]

    params['log_dir'] = path.join(params['base_dir'],'logs')
    params['pipeline_config_dir'] = path.join(params['base_dir'],'config')

    setup = pipeline_setup.pipeline_setup(params)

    return setup


def get_datasets_for_reduction(setup,log):
    """Function to compose a list of the datasets to be reduced.

    Options:
    1) If a file reduce_datasets.txt exists within the proc/configs directory
    this will be read.  This file should contain a list (one per line) of the
    reduction sub-directory names to be reduced, without paths.

    2) If no reduce_datasets.txt file is found the code returns a list of
    all of the dataset sub-directories found in the /proc directory.
    """

    datasets_file = path.join(setup.pipeline_config_dir,'reduce_datasets.txt')

    if path.isfile(datasets_file) == True:

        log.info('Found a reduce_datasets instruction file')

        file_lines = open(datasets_file).readlines()

        datasets = []

        log.info('Going to reduce the following datasets:')

        for line in file_lines:

            if len(line.replace('\n','')) > 0:
                datasets.append(line.replace('\n',''))

            log.info(datasets[-1])

    else:

        log.info('No instruction file found, going to reduce all datasets')

        dir_list = glob.glob(path.join(setup.base_dir,'*'))

        datasets = []

        for item in dir_list:

            if 'logs' not in item and 'config' not in item \
                and len(path.basename(item)) > 0:

                datasets.append(path.basename(item))

                log.info(datasets[-1])

    return datasets


def run_reductions(setup,log,datasets):
    """Function to trigger the reduction of one or more datasets.

    Inputs:
        setup       PipelineSetup object
        datasets    list                    Dataset red_dir names
    """

    log.info('Starting reductions:')

    for data_dir in datasets:

        dataset_dir = path.join(setup.base_dir,data_dir)

        pid = trigger_reduction(setup,dataset_dir,debug=False)

        log.info(' -> Dataset '+path.basename(dataset_dir)+\
                ' reduction PID '+str(pid))

def trigger_reduction(setup,dataset_dir,debug=False):
    """Function to spawn a child process to run the reduction of a
    single dataset.

    Inputs:
        setup       PipelineSetup object
        dataset_dir   str    Path to dataset red_dir
    """

    if debug == False:

        command = path.join(setup.software_dir,'reduction_control.py')

    else:

        if 'tests' in setup.software_dir:

            command = path.join(setup.software_dir,'counter.py')

        else:

            command = path.join(setup.software_dir,'tests','counter.py')

    args = ['python', command, dataset_dir, setup.phot_db_path, setup.red_mode]

    p = subprocess.Popen(args, stdout=subprocess.PIPE)

    return p.pid



if __name__ == '__main__':

    pipeline_control()
