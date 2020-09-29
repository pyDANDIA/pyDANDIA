# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 11:01:19 2017

@author: rstreet
"""
from os import getcwd, path, remove, mkdir
from sys import argv, exit
from sys import path as systempath
import copy
import glob
import subprocess
from datetime import datetime
from pyDANDIA import pipeline_setup
from pyDANDIA import stage0
from pyDANDIA import stage1
from pyDANDIA import stage2
from pyDANDIA import reference_astrometry
from pyDANDIA import stage3
from pyDANDIA import stage3_db_ingest
from pyDANDIA import stage4
from pyDANDIA import stage5
from pyDANDIA import stage6
from pyDANDIA import logs
from pyDANDIA import reset_stage_metadata
from pyDANDIA import config_utils
from pyDANDIA import metadata
from pyDANDIA import phot_db as db_phot
from pyDANDIA import image_handling
from pyDANDIA import lightcurves

def reduction_control():
    """Main driver function for the pyDANDIA pipelined reduction of an
    individual dataset.

    A single dataset is defined as a set of images taken with a consistent
    instrument configuration and filter of a single sky pointing.

    Input parameters:
        dataset_red_dir   str   Full path to the reduction directory for this
                                dataset
    """

    reduction_version = 'reduction_control v0.3'

    (setup,params,proc_log_dir) = get_args()

    red_log = logs.start_pipeline_log(proc_log_dir, 'reduction_control',
                                  version=reduction_version)

    red_log.info('Pipeline setup: '+setup.summary()+'\n')

    if setup.red_mode == 'data_preparation':

        run_data_preparation(setup,red_log,select_ref=True)

    elif setup.red_mode == 'added_data_preparation':

        run_data_preparation(setup,red_log,select_ref=False)

    elif setup.red_mode == 'reference_analysis':

        run_reference_image_analysis(setup,red_log)

    elif setup.red_mode == 'image_analysis':

        run_image_analysis(setup,red_log,params)

    elif setup.red_mode == 'stage3_db_ingest':

        run_stage3_db_ingest(setup,red_log,params)

    elif setup.red_mode == 'stage6':

        run_stage6_db_ingest(setup,red_log,params)

    elif setup.red_mode == 'auto':

        run_automatic_reduction(setup,red_log,params)

    else:
        red_log.info('ERROR: unrecognised reduction mode ('+setup.red_mode+') selected')

    logs.close_log(red_log)

def run_data_preparation(setup,red_log=None,select_ref=False):
    """Function to run in sequence stages 0 - 2 for a single dataset"""

    if red_log!=None:
        red_log.info('Pipeline setup: '+setup.summary()+'\n')

    (status,report,meta_data) = stage0.run_stage0(setup)

    if red_log!=None:
        red_log.info('Completed stage 0 with status '+repr(status)+': '+report)

    status = execute_stage(stage1.run_stage1, 'stage 1', setup, status, red_log)

    if select_ref:
        status = execute_stage(stage2.run_stage2, 'stage 2', setup, status, red_log)

def run_reference_image_analysis(setup,red_log):
    """Function to run the pipeline stages which perform the analysis of a
    reference image in sequence."""

    red_log.info('Pipeline setup: '+setup.summary()+'\n')

    status = 'OK'

    status = execute_stage(reference_astrometry.run_reference_astrometry,
                           'reference astrometry', setup, status, red_log)

    status = execute_stage(stage3.run_stage3, 'stage 3', setup, status, red_log)

def run_image_analysis(setup,red_log,params):
    """Function to run the sequence of stages which perform the image
    subtraction and photometry for a dataset"""

    red_log.info('Pipeline setup: '+setup.summary()+'\n')

    status = 'OK'

    status = execute_stage(stage4.run_stage4, 'stage 4', setup, status, red_log)

    status = execute_stage(stage5.run_stage5, 'stage 5', setup, status, red_log)

    status = execute_stage(stage6.run_stage6, 'stage 6', setup, status, red_log, **params)

def run_stage3_db_ingest_bulk(setup,red_log,params):
    """Function to run stage3_db_ingest for a set of datasets read from a file
    File format is one dataset per line plus a column indicating whether or
    not a given dataset is the primary reference, i.e.:
    /path/to/data/red/dir  primary_ref
    /path/to/data/red/dir  not_ref
    /path/to/data/red/dir  not_ref
    ...
    """

    datasets = parse_dataset_list(params['data_file'])

    for dataset,ref_flag in datasets.items():

        dparams = copy.copy(params)
        dparams['red_dir'] = dataset
        dparams['log_dir'] = path.join(dparams['red_dir'],'..','logs')
        dparams['base_dir'] = path.join(dparams['red_dir'],'..')

        dsetup = pipeline_setup.pipeline_setup(dparams)

        if 'primary_ref' in ref_flag or 'primary-ref' in ref_flag:
            red_log.info('Ingesting '+path.basename(dataset)+' as the primary reference dataset')

            (status,report) = stage3_db_ingest.run_stage3_db_ingest(dsetup, primary_ref=True)

        else:

            red_log.info('Ingesting '+path.basename(dataset))

            (status,report) = stage3_db_ingest.run_stage3_db_ingest(dsetup, primary_ref=False)

        red_log.info('Completed stage3_db_ingest for '+path.basename(dataset)+' with status '+repr(status))
        red_log.info(repr(report))

def run_stage3_db_ingest(setup, red_log, params):
    """Function to run stage3_db_ingest for a single dataset. A flag indicates
    whether or not the given dataset is the primary reference.
    """

    if 'primary' in params['primary_flag']:
        red_log.info('Ingesting '+path.basename(setup.red_dir)+' as the primary reference dataset')

        (status,report) = stage3_db_ingest.run_stage3_db_ingest(setup, primary_ref=True)

    else:

        red_log.info('Ingesting '+path.basename(setup.red_dir))

        (status,report) = stage3_db_ingest.run_stage3_db_ingest(setup, primary_ref=False)

    red_log.info('Completed stage3_db_ingest for '+str(path.basename(setup.red_dir))+' with status '+repr(status))
    red_log.info(repr(report))

def parse_dataset_list(file_path):

    if path.isfile(file_path) == False:
        raise IOError('Cannot find list of datasets')

    flines = open(file_path).readlines()

    datasets = {}
    for line in flines:
        entries = line.replace('\n','').split()
        datasets[entries[0]] = entries[1]

    return datasets

def run_stage6_db_ingest_bulk(setup,red_log,params):
    """Function to run stage6 including the DB ingest for a set of datasets read from a file
    File format is one dataset per line plus a column indicating whether or
    not a given dataset is the primary reference, i.e.:
    /path/to/data/red/dir  primary_ref
    /path/to/data/red/dir  not_ref
    /path/to/data/red/dir  not_ref
    ...
    """

    datasets = parse_dataset_list(params['data_file'])

    for dataset,ref_flag in datasets.items():

        dparams = copy.copy(params)
        dparams['red_dir'] = dataset
        dparams['log_dir'] = path.join(dparams['red_dir'],'..','logs')
        dparams['base_dir'] = path.join(dparams['red_dir'],'..')

        dsetup = pipeline_setup.pipeline_setup(dparams)

        red_log.info('Ingesting '+path.basename(dataset))

        (status,report) = stage6.run_stage6(dsetup)

def run_stage6_db_ingest(setup,red_log,params):
    """Function to run stage6 including the DB ingest for a set of datasets read from a file
    File format is one dataset per line plus a column indicating whether or
    not a given dataset is the primary reference, i.e.:
    /path/to/data/red/dir  primary_ref
    /path/to/data/red/dir  not_ref
    /path/to/data/red/dir  not_ref
    ...
    """

    red_log.info('Ingesting '+path.basename(setup.red_dir))

    (status,report) = stage6.run_stage6(setup)

def get_auto_config(setup,log):
    """Function to read in the configuration file for automatic reductions"""

    config_file = path.join(setup.pipeline_config_dir, 'auto_pipeline_config.json')

    config = config_utils.build_config_from_json(config_file)

    boolean_keys = ['use_gaia_phot', 'catalog_xmatch', 'build_phot_db']
    for key in boolean_keys:
        if key in config.keys():
            if 'true' in str(config[key]).lower():
                config[key] = True
            else:
                config[key] = False

    log.info('Read in configuration for automatic reductions:')

    for key, value in config.items():
        log.info(key+': '+repr(value))

    return config

def run_automatic_reduction(setup,red_log,params):
    """Function to run an automatic reduction of a single dataset"""

    red_log.info('Starting automatic reduction of '+path.basename(setup.red_dir))

    config = get_auto_config(setup,red_log)
    config['primary_flag'] = params['primary_flag']

    locked = check_dataset_lock(setup,red_log)

    if not locked:
        lock_dataset(setup,red_log)

        existing_reduction = got_existing_reference(setup,red_log)

        if existing_reduction:
            status = run_existing_reduction(setup, config, red_log)

        else:
            status = run_new_reduction(setup, config, red_log)

        unlock_dataset(setup,red_log)

def run_existing_reduction(setup, config, red_log):
    """Function to reduce a dataset with existing data that has been
    reduced at least once successfully already.  This assumes that a reference
    image has been selected and reduced, and the reference photometry stored
    in the corresponding photometric DB."""

    (status,report,meta_data) = stage0.run_stage0(setup)
    red_log.info('Completed stage 0 with status '+repr(status)+': '+report)

    status = execute_stage(stage1.run_stage1, 'stage 1', setup, status, red_log)

    status = execute_stage(stage4.run_stage4, 'stage 4', setup, status, red_log)

    reset_stage_metadata.reset_red_status_for_stage(setup.red_dir,5)
    red_log.info('Reset stage 5 of existing reduction')

    status = execute_stage(stage5.run_stage5, 'stage 5', setup, status, red_log)

    sane = check_stage3_db_ingest(setup,red_log)

    if sane == False and config['build_phot_db']:
        run_stage3_db_ingest(setup,red_log,config)

    status = execute_stage(stage6.run_stage6, 'stage 6', setup, status, red_log, **config)

    extract_target_lightcurve(setup, red_log)

    return status

def run_new_reduction(setup, config, red_log):

    (status,report,meta_data) = stage0.run_stage0(setup)
    red_log.info('Completed stage 0 with status '+repr(status)+': '+report)

    status = execute_stage(stage1.run_stage1, 'stage 1', setup, status, red_log, **config)

    if not check_for_assigned_ref_image(setup, red_log):
        status = execute_stage(stage2.run_stage2, 'stage 2', setup, status, red_log, **config)

    status = execute_stage(reference_astrometry.run_reference_astrometry,
                           'reference astrometry', setup, status, red_log, **config)

    status = execute_stage(stage3.run_stage3, 'stage 3', setup, status, red_log, **config)

    status = execute_stage(stage4.run_stage4, 'stage 4', setup, status, red_log)

    status = execute_stage(stage5.run_stage5, 'stage 5', setup, status, red_log)

    if config['build_phot_db']:
        run_stage3_db_ingest(setup,red_log,config)

    status = execute_stage(stage6.run_stage6, 'stage 6', setup, status, red_log, **config)

    extract_target_lightcurve(setup, red_log)

    return status

def check_for_assigned_ref_image(setup, log):
    """Function to check whether a reduction has been assigned a reference
    image, or needs stage 2 to be run."""

    log.info('Testing whether a reference image has been assigned for this reduction')

    reduction_metadata = metadata.MetaData()
    if path.join(setup.red_dir,'pyDANDIA_metadata.fits') == False:
        return False

    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                                  'pyDANDIA_metadata.fits',
                                                  'data_architecture' )

    if 'REF_IMAGE' in reduction_metadata.data_architecture[1].keys():
        log.info('Reference image assigned: '+reduction_metadata.data_architecture[1]['REF_IMAGE'][0])
        return True
    else:
        log.info('No reference image assigned to this reduction')
        return False

def extract_target_lightcurve(setup, log):
    """Function to extract the lightcurve of the target indicated in the
    FITS image header."""

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                              'pyDANDIA_metadata.fits',
                                              'data_architecture' )

    ref_path = str(reduction_metadata.data_architecture[1]['REF_PATH'][0]) +'/'+\
                str(reduction_metadata.data_architecture[1]['REF_IMAGE'][0])

    ref_header = image_handling.get_science_header(ref_path)

    lc_dir = path.join(setup.red_dir, 'lc')
    if path.isdir(lc_dir) == False:
        mkdir(lc_dir)

    ra = ref_header['CAT-RA']
    dec = ref_header['CAT-DEC']
    if 'N/A' in ra or 'N/A' in dec:
        ra = ref_header['RA']
        dec = ref_header['DEC']

    # All coordinates and radius must be in decimal degrees
    params = {'red_dir': setup.red_dir, 'db_file_path': setup.phot_db_path,
                'ra': ra, 'dec': dec,
                'radius': (2.0 / 3600.0), 'output_dir': lc_dir }

    log.info('Searching phot DB '+setup.phot_db_path+' for '+ref_header['OBJECT'])

    lightcurves.extract_star_lightcurve_isolated_reduction(params, log=log, format='csv')

    log.info('Extracted lightcurve for '+ref_header['OBJECT']+' at RA,Dec='+\
            repr(ref_header['CAT-RA'])+', '+repr(ref_header['CAT-DEC'])+\
            ' and output to '+lc_dir)

def check_stage3_db_ingest(setup,log):
    """Function to verify whether the photometry for a dataset has been
    ingested into the corresponding photometric database."""

    sane = False

    if path.isfile(setup.phot_db_path):

        conn = db_phot.get_connection(dsn=setup.phot_db_path)
        conn.execute('pragma synchronous=OFF')

        reduction_metadata = metadata.MetaData()
        reduction_metadata.load_a_layer_from_file( setup.red_dir,
                                                  'pyDANDIA_metadata.fits',
                                                  'data_architecture' )
        ref_filename = str(reduction_metadata.data_architecture[1]['REF_IMAGE'][0])

        log.info('Checking phot DB '+str(setup.phot_db_path)+' for photometry from dataset reference image '+ref_filename)

        query = 'SELECT refimg_id, filename FROM reference_images WHERE filename ="' + ref_filename + '"'
        refimage = db_phot.query_to_astropy_table(conn, query, args=())

        if len(refimage) == 0:
            sane = False
        else:
            sane = True

            log.info(' -> '+repr(sane))

    else:
        log.info('No phot DB found for this field at '+str(setup.phot_db_path))

    conn.close()

    return sane

def check_dataset_lock(setup,log):
    """Function to check for a lockfile in a given dataset before starting
    a reduction"""

    lockfile = path.join(setup.red_dir,'dataset.lock')

    if path.isfile(lockfile) == True:
        log.info(path.basename(setup.red_dir)+' is locked')
        status = True
    else:
        log.info(path.basename(setup.red_dir)+' is not locked')
        status = False

    return status

def lock_dataset(setup,log):
    """Function to set a lock on a dataset to prevent simultaneous reductions"""

    lockfile = path.join(setup.red_dir,'dataset.lock')
    ts = datetime.utcnow()

    f = open(lockfile,'w')
    f.write(ts.strftime('%Y-%m-%dT%H:%M:%S'))
    f.close()

    log.info('-> Locked dataset '+path.basename(setup.red_dir))

def unlock_dataset(setup,log):
    """Function to remove a lock on a dataset once reductions have completed"""

    lockfile = path.join(setup.red_dir,'dataset.lock')

    if path.isfile(lockfile):
        remove(lockfile)
        log.info('-> Unlocked dataset '+path.basename(setup.red_dir))

    else:
        log.info('-> WARNING dataset '+path.basename(setup.red_dir)+' found unlocked when lock expected')

def got_existing_reference(setup,log):
    """Function to check whether a dataset has already been reduced i.e.
    whether there is an existing reference image"""

    existing_reduction = False

    if path.join(setup.red_dir, 'ref'):

        list_red_ref_image = glob.glob(path.join(setup.red_dir, 'ref', '*_res.fits'))

        if len(list_red_ref_image) > 0:

            existing_reduction = True

            log.info('Found a reduced reference image for dataset '+path.basename(setup.red_dir))

        else:
            log.info('Found ref sub-directory but no reduced reference image for dataset '+path.basename(setup.red_dir))

    else:
        log.info('Found no pre-existing reduction for dataset '+path.basename(setup.red_dir))

    return existing_reduction

def execute_stage(run_stage_func, stage_name, setup, status, red_log, **kwargs):
    """Function to execute a stage and verify whether it completed successfully
    before continuing.

    Accepts as an argument the status of the previous stage in order to check
    whether or not to continue the reduction.  If the reduction proceeds, this
    status is overwritten with the status output of the next stage.

    Inputs:
        :param function run_stage_func: Single imported function
        :param string stage_name: Function name, for logging output
        :param object setup: pipeline setup object instance
        :param string status: Status of execution of the most recent stage
        :param logging log: open log file object

    Outputs:
        :param string status: Status of execution of the most recent stage
    """

    if 'OK' in status:

        if '0' in stage_name:

            (status, report, metadata) = run_stage_func(setup)

        else:

            (status, report) = run_stage_func(setup, **kwargs)

        red_log.info('Completed '+stage_name+' with status '+\
                    repr(status)+': '+report)

    if 'OK' not in status:

        red_log.info('ERROR halting reduction due to previous errors')

        logs.close_log(red_log)

        exit()

    return status

def parallelize_stages345(setup, status, red_log):
    """Function to execute stages 4 & 5 in parallel with stage 3.

    Inputs:
        :param object setup: pipeline setup object instance
        :param string status: Status of execution of the most recent stage
        :param logging log: open log file object

    Outputs:
        :param string status: Status of execution of the most recent stage
    """

    red_log.info('Executing stage 3 in parallel with stages 4 & 5')

    process3 = trigger_stage_subprocess('stage3',setup,re_log,wait=False)

    process4 = trigger_stage_subprocess('stage4',setup,red_log,wait=True)
    process5 = trigger_stage_subprocess('stage5',setup,red_log,wait=True)

    red_log.info('Completed stages 4 and 5; now waiting for stage 3')

    (outs, errs) = process3.communicate()

    if errs == None:

        process3.wait()
        red_log.info('Completed stage 3')

    else:

        red_log.info('ERROR: Problem encountered in stage 3:')
        red_log.info(errs)

    red_log.info('Completed parallel stages')

    return 'OK'

def trigger_stage_subprocess(stage_code,setup,red_log,wait=True):
    """Function to run a stage as a separate subprocess

    Inputs:
        :param string stage_code: Stage to be run without spaces, e.g. stage0
        :param object setup: Pipeline setup instance
    """

    command = path.join(setup.software_dir,'run_stage.py')

    args = ['python', command, stage_code, setup.red_dir]

    p = subprocess.Popen(args, stdout=subprocess.PIPE)

    red_log.info('Started '+stage_code+', PID='+str(p.pid))

    if wait:

        red_log.info('Waiting for '+stage_code+' to finish')

        p.wait()

        red_log.info('Completed '+stage_code)

    return p

def get_args():
    """Function to obtain the command line arguments necessary to run a
    single-dataset reduction."""

    helptext = """
                    pyDANDIA Reduction Control

    Main driver program to run pyDANDIA in pipeline mode for a single dataset.

    Command and options:
    > python reduction_control.py red_dir_path phot_db_path mode primary_flag [-v N ]

    where red_dir_path is the path to a dataset's reduction directory
          phot_db_path is the path to a photometry database
          mode is the mode of reduction required
          primary_flag indicates whether the dataset is to be used as a primary reference or not {primary_ref, non_ref}

    Reduction mode options are:
          mode  new_reference

    The -v flag controls the verbosity of the pipeline logging output.  Values
    N can be:
    -v 0 [Default] Essential logging output only, written to log file.
    -v 1           Detailed logging output, written to log file.
    -v 2           Detailed logging output, written to screen and to log file.

    To display information on options:
    > python reduction_control.py -help
    """

    if '-help' in argv:
        print(helptext)
        exit()

    reduction_modes = ['data_preparation',
                       'added_data_preparation',
                       'reference_analysis',
                       'image_analysis',
                       'stage3_db_ingest',
                       'stage6',
                       'auto']

    params = {}

    if len(argv) == 1:

        params['red_dir'] = input('Please enter the path to the datasets reduction directory: ')
        params['db_file_path'] = input('Please enter the path to the photometric database: ')
        params['mode'] = input('Please enter the reduction mode, one of {'+','.join(reduction_modes)+'}: ')
        params['primary_flag'] = input('Please input reference status flag for this dataset {primary_ref, non_ref}: ')

    else:

        params['red_dir'] = argv[1]
        params['db_file_path'] = argv[2]
        params['mode'] = argv[3]
        params['primary_flag'] = argv[4]

    if '-v' in argv:

        idx = argv.index('-v')

        if len(argv) >= idx + 1:

            params['verbosity'] = int(argv[idx+1])

    if '-no-phot-db' in argv:
        params['build_phot_db'] = False
    else:
        params['build_phot_db'] = True

    params['log_dir'] = path.join(params['red_dir'],'..','logs')
    proc_log_dir = path.join(params['red_dir'],'logs')
    params['pipeline_config_dir'] = path.join(params['red_dir'],'..','config')
    params['base_dir'] = path.join(params['red_dir'],'..')
    params['software_dir'] = getcwd()

    setup = pipeline_setup.pipeline_setup(params)
    setup.red_mode = params['mode']

    return setup, params, proc_log_dir



if __name__ == '__main__':
    reduction_control()
