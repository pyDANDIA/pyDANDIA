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
import copy
from pyDANDIA import pipeline_setup
import glob
import subprocess
from pyDANDIA import logs

def pipeline_control():
    """Main driver program controlling the reduction of multiple datasets
    with pyDANDIA.
    """

    pipeline_version = 'pipeline_control v0.2'

    (setup,kwargs) = get_args()

    log = logs.start_pipeline_log(setup.log_dir, 'pipeline_control',
                               version=pipeline_version)
    log.info(setup.summary())

    datasets = get_datasets_for_reduction(setup,log)

    run_reductions(setup,log,datasets,kwargs)

    logs.close_log(log)


def get_args():
    """Function to acquire the necessary commandline arguments to run
    pyDANDIA in pipeline mode."""

    params = {}
    kwargs = {}

    if len(argv) == 1:
        params['base_dir'] = input('Please enter the path to the base directory: ')
        params['phot_db_path'] = input('Please enter the path to the database file [or None to switch off DB]: ')
        print('''Please enter the required reduction mode out of:
        {data_preparation, added_data_preparation, reference_analysis, image_analysis, stage3_db_ingest, stage6, stage3, post_processing}''')
        params['red_mode'] = input('Reduction mode: ')

    else:
        params['base_dir'] = argv[1]
        params['phot_db_path'] = argv[2]
        params['red_mode'] = argv[3]

        for a in argv:
            if 'software=' in a:
                params['software_dir'] = str(a).split('=')[-1]
            if 'python=' in a:
                kwargs['python'] = str(a).split('=')[-1]

    params['log_dir'] = path.join(params['base_dir'],'logs')
    params['pipeline_config_dir'] = path.join(params['base_dir'],'config')
    params['verbosity'] = 0

    setup = pipeline_setup.pipeline_setup(params)
    setup.phot_db_path = params['phot_db_path']

    if 'None' in params['phot_db_path']:
        kwargs['build_phot_db'] = False

    return setup, kwargs


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

        datasets = {}

        log.info('Going to reduce the following datasets:')

        for line in file_lines:

            if len(line.replace('\n','')) > 0:
                (dataset_code, ref_status) = line.replace('\n','').split()
                datasets[dataset_code] = ref_status

            log.info(dataset_code)

    else:

        raise IOError('Cannot find input list of datasets.  Looking for '+datasets_file)

    return datasets


def run_reductions(setup,log,datasets,kwargs):
    """Function to trigger the reduction of one or more datasets.

    Inputs:
        setup       PipelineSetup object
        datasets    list                    Dataset red_dir names
    """

    if kwargs['build_phot_db'] and setup.red_mode in ['stage3_db_ingest', 'stage6']:

        log.info('Starting sequential reductions')

        primary = None
        for data_dir,data_status in datasets.items():
            if 'primary' in data_status and primary == None:
                primary = data_dir
            elif 'primary' in data_status and primary != None:
                raise TypeError('Multiple primary reference datasets indicated in input file')

        data_order = []
        if primary != None:

            dsetup = setup.duplicate()
            dsetup.red_dir = path.join(dsetup.base_dir,primary)
            dsetup.log_dir = path.join(dsetup.base_dir,primary)

            log.info('Running '+dsetup.red_mode+' for '+primary+' as primary reference')
            log.info(dsetup.summary())

            trigger_single_reduction(dsetup, primary, 'primary_ref')

        for data_dir,data_status in datasets.items():
            if data_dir != primary:
                data_order.append(data_dir)

        for data_dir in data_order:
            dsetup = setup.duplicate()
            dsetup.red_dir = path.join(dsetup.base_dir,data_dir)
            dsetup.log_dir = path.join(dsetup.base_dir,data_dir)

            log.info('Running '+dsetup.red_mode+' for '+data_dir+' as standard dataset')
            log.info(dsetup.summary())

            trigger_single_reduction(dsetup, data_dir, 'non_ref')

    else:
        log.info('Starting parallel reductions:')

        for data_dir,data_status in datasets.items():

            dataset_dir = path.join(setup.base_dir,data_dir)

            pid = trigger_parallel_reduction(setup,dataset_dir,data_status,kwargs,log,debug=False)

            log.info(' -> Dataset '+path.basename(dataset_dir)+\
                    ' reduction PID '+str(pid))

def trigger_parallel_reduction(setup,dataset_dir,data_status,kwargs,log,debug=False):
    """Function to spawn a child process to run the reduction of a
    single dataset.

    Inputs:
        setup       PipelineSetup object
        dataset_dir   str    Path to dataset red_dir
    """

    if 'python' in kwargs.keys():
        pythonpath = kwargs['python']
    else:
        pythonpath = 'python'
    log.info('Using pythonpath: '+pythonpath)

    if 'tests' in setup.software_dir:

            command = path.join(setup.software_dir,'counter.py')
            args = args = [pythonpath, command, dataset_dir, setup.phot_db_path, setup.red_mode]

    elif setup.red_mode in ['data_preparation', 'added_data_preparation',
                            'reference_analysis', 'image_analysis']:

        command = path.join(setup.software_dir,'reduction_control.py')
        args = [pythonpath, command, dataset_dir, setup.phot_db_path, setup.red_mode, data_status]

        if 'build_phot_db' in kwargs.keys() and kwargs['build_phot_db']==False:
            args += ['-no-phot-db']

    elif setup.red_mode in ['stage3']:

        command = path.join(setup.software_dir,'run_stage.py')
        args = [pythonpath, command, setup.red_mode, dataset_dir, setup.phot_db_path]

    elif setup.red_mode in ['stage6'] and kwargs['build_phot_db'] == False:

        command = path.join(setup.software_dir,'run_stage.py')
        args = [pythonpath, command, setup.red_mode, dataset_dir, setup.phot_db_path]

    elif setup.red_mode == 'post_processing':

        command = path.join(setup.software_dir,'run_stage.py')
        args = [pythonpath, command, setup.red_mode, dataset_dir, setup.phot_db_path]

    else:
        raise ValueError('Reduction mode '+str(setup.red_mode)+' not yet supported in parallel mode')

    p = subprocess.Popen(args, stdout=subprocess.PIPE)

    return p.pid


def trigger_single_reduction(setup,dataset_dir,data_status):
    """Function to spawn a child process to run the reduction of a
    single dataset.

    Inputs:
        setup       PipelineSetup object
        dataset_dir   str    Path to dataset red_dir
    """

    if setup.red_mode == 'stage3_db_ingest' or setup.red_mode == 'stage6':

        command = path.join(setup.software_dir,'reduction_control.py')
        args = ['python', command, setup.red_dir, setup.phot_db_path, setup.red_mode, data_status, '-v', '0']

        pid = subprocess.call(args, stdout=subprocess.PIPE)

    else:
        raise TypeError('Can only trigger a single reduction for stage3_db_ingest or stage6')

    return pid


if __name__ == '__main__':

    pipeline_control()
