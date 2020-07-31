from os import getcwd, path, remove
from sys import argv
from sys import path as systempath
cwd = getcwd()
systempath.append(path.join(cwd,'../'))
import copy
from pyDANDIA import pipeline_setup
import glob
import subprocess
from pyDANDIA import pipeline_setup
from pyDANDIA import config_utils
from pyDANDIA import logs
from datetime import datetime

class DataGroup:

    def __init__(self):
        self.target = None
        self.group_id = None
        self.aperture_class = None
        self.dir_list = []
        self.phot_db = None
        self.primary_ref_dir = None
        self.ok_to_process = True
        self.pid_list = []

    def find_primary_ref(self, log):

        primary_ref = self.primary_ref_dir

        for dir in self.dir_list:
            if self.primary_ref_dir == None and path.isfile(path.join(dir,'primary_ref.flag')):
                self.set_primary_ref(dir,log)

            elif self.primary_ref != None and path.isfile(path.join(dir,'primary_ref.flag')):
                log.info('-> Primary reference set to '+self.primary_ref+' and found primary ref flag in '+dir)

        log.info('-> Using '+self.primary_ref+' as the primary reference dataset for '+self.target)

        return self.primary_ref_dir

    def set_primary_ref(self,primary_ref_dir,log):

        self.primary_ref_dir = primary_ref_dir

        flag = path.join(self.primary_ref_dir,'primary_ref.flag')

        if flag == False:
            f = open(flag,'w')
            ts = datetime.utcnow()
            f.write(ts.strftime('%Y-%m-%dT%H:%M:%S'))
            f.close()

        log.info('-> Set primary reference for '+self.target+' to '+self.primary_ref_dir)

    def choose_primary_ref(self, log):
        """Attempt to automatically select the best dataset from a group to act as
        the primary reference dataset for the field, based on telescope aperture,
        filter and site.  1m0 datasets are preferred, as they provide the best
        balance of field of view and limiting magnitude."""

        filter_preference = ['ip', 'rp', 'gp']
        aperture_preference = ['1m0', '2m0', '0m4']
        site_preference = ['lsc', 'ogg', 'cpt', 'elp', 'tfn', 'coj']
        if len(self.dir_list) == 1:
            self.primary_ref_dir = self.dir_list[0]

        else:
            selection = self.dir_list
            selection = choose_dir_from_options(selection,aperture_preferences,'aperture',log)
            selection = choose_dir_from_options(selection,filter_preferences,'filter',log)
            selection = choose_dir_from_options(selection,site_preferences,'site',log)

            if len(selection) == 1:
                self.set_primary_ref(selection[0],log)

            else:
                log.info('ERROR: Multiple possible options for primary reference dataset')
                self.ok_to_process = False

        return self.primary_ref_dir

    def choose_dir_from_options(options,preferences,criterion,log):

        selection = []
        for pref in preference:

            for opt in options:
                if pref in path.basename(opt):
                    selection.append(opt)

        log.info('Selected the following options based on preferences for '+criterion)
        log.info(repr(selection))

        return selection

def reduce_datasets(config,datasets,log):

    log.info('Starting reduction subprocesses')

    pid_list = []

    for dir in datasets:

        data_status = 'primary-ref'
        phot_db_path = path.join(dir, path.basename(dir)+'_phot.db')

        pid = trigger_parallel_auto_reduction(config,dir,phot_db_path,
                                                data_status)

        pid_list.append(pid)

        log.info(' -> Dataset '+path.basename(dir)+\
                ' reduction PID '+str(pid))

        if len(pid_list) >= config['group_processing_limit']:
            log.info('WARNING: Reached maximum number of parallel reduction processes')
            break

    return pid_list

def run_pipeline():

    config = get_config()

    log = logs.start_pipeline_log(config['log_dir'], 'automatic_pipeline')

    # Identify datasets to be reduced
    datasets = parse_configured_datasets(config,log)
    datasets = sanity_check_data_before_reduction(datasets,log)

    # Start reductions:
    pid_list = reduce_datasets(config, datasets, log)

    #exit_codes = [p.wait() for p in all_processes]

    logs.close_log(log)

def get_config():
    """Function to acquire the necessary commandline arguments to run
    pyDANDIA in automatic mode."""

    if len(argv) == 2:
        config_file = argv[1]

        config = config_utils.build_config_from_json(config_file)
    else:
        exit()

    return config

def list_supported_instruments(config,log):

    instruments = []

    config_files = glob.glob(path.join(config['config_dir'],'inst_config_*.json'))

    for f in config_files:
        instruments.append( f.remove('.json','').split('_')[-1] )

    log.info('Found instrument configurations for '+str(len(instruments))+' instruments')

    return instruments

def check_instrument_supported(instruments,data_dir):

    instrument = path.basename(data_dir).split('_')[1].split('-')[-1]

    if instrument in instruments:
        return True
    else:
        return False

def identify_dataset_groups(config, instruments, log):

    subdirs = glob.glob(path.join(config['data_red_dir'],'*'))

    data_groups = {}

    for dir in subdirs:

        # Check that the subdir is actually a dataset reduction directory
        # Also check that it's for a supported instrument
        if path.isdir(dir) and path.isdir(path.join(dir,'data')) and \
            check_instrument_supported(instruments,dir):

            target = path.basename(dir).split('_')[0]
            ap_class = path.basename(dir).split('_')[1].split('-')[2]

            group_id = target+'_'+ap_class

            phot_db_path = get_phot_db_path(config,target)

            if group_id in data_groups.keys():
                dg = data_groups[group_id]
                if path.isfile(phot_db_path):
                    dg.phot_db_path = phot_db_path
                dg.dir_list.append(dir)
                data_groups[group_id] = dg
            else:
                dg = DataGroup()
                dg.target = target
                dp.aperture_class = ap_class
                dg.group_id = group_id
                if path.isfile(phot_db_path):
                    dg.phot_db_path = phot_db_path
                dg.dir_list.append(dir)
                data_groups[group_id] = dg

        elif path.isdir(dir) and path.isdir(path.join(dir,'data')) and \
            check_instrument_supported(instruments,dir) == False:

            log.info('WARNING: No instrument configuration available for dataset '+dir)

    if len(data_groups) == 0:
        log.info('Found no data to reduce')

    else:
        log.info('Identifed the following groups of datasets per field to be reduced: ')

        for target, dg in data_groups.items():
            log.info(target+': ')
            for dir in dg.dir_list:
                log.info('    '+dir)

    return data_groups

def identify_primary_reference_datasets(config,data_groups,log):

    for target in data_groups.keys():
        dg = data_groups[target]

        if dg.phot_db_path == None:
            dg.phot_db_path = get_phot_db_path(config,target)

            # Look for a primary reference data flag, if any, and select one
            # if none found:
            primary_ref = dg.find_primary_ref(log)

            if primary_ref == None:
                primary_ref = dg.choose_primary_ref(log)

        # If a photometry DB exists, then a primary reference must also exist.
        # If not, raise an error:
        else:
            primary_ref = dg.find_primary_ref(log)

            if primary_ref == None:
                log.info('ERROR: No primary reference could be identified for '+target+' but phot_db exists.')
                dg.ok_to_process = False

        data_groups[target] = dg

    return data_groups

def parse_configured_datasets(config,log):
    """Function to identify the datasets to be reduced based on the configuration"""

    datasets = []

    if '@' in config['reduce_datasets']:
        dataset_list = config['reduce_datasets'].replace('@','')

        if path.isfile(dataset_list):

            entries = open(dataset_list,'r').readlines()
            for dataset_name in entries:
                if len(dataset_name.replace('\n','')) > 0:
                    datasets.append( path.join(config['data_red_dir'], dataset_name.replace('\n','')) )

            log.info('Read '+str(len(datasets))+' datasets to be reduced from '+dataset_list+':')
            [log.info(path.basename(x)) for x in datasets]

        else:
            message = 'ERROR: Cannot find configured list of datasets to be reduced, '+dataset_list
            log.info(message)
            raise IOError(message)

    elif str(config['reduce_datasets']).upper() == 'ALL':

        entries = glob.glob(path.join(config['data_red_dir'],'*'))

        for dataset_path in entries:
            if path.isdir(dataset_path) and path.isdir(path.join(dataset_path,'data')):
                datasets.append( dataset_path )

        log.info('Found '+str(len(datasets))+' datasets to be reduced in '+config['data_red_dir']+':')
        [log.info(path.basename(x)) for x in datasets]

    return datasets

def sanity_check_data_before_reduction(datasets,log):
    """This function reviews the lock status of each dataset and performs basic sanity checks,
    such as whether the data directory exists, before adding a dataset to the list to
    be reduced."""

    sane_datasets = []

    log.info('Pre-reduction checks of datasets to be reduced:')

    for dir in datasets:
        log.info('Checking '+dir+':')
        dir_status = check_dataset_dir_structure(dir, log)
        lock_status = check_dataset_dir_unlocked(dir,log)

        if dir_status and lock_status:
            sane_datasets.append( dir )
            log.info(' -> Accepted for reduction')

    return sane_datasets

def check_dataset_dir_structure(dataset_path, log):

    if path.isdir(dataset_path) and path.isdir(path.join(dataset_path,'data')):
        log.info(' -> Directory structure looks OK')
        return True
    else:
        log.info(' -> ERROR: Wrong directory structure for pyDANDIA')
        return False

def check_dataset_dir_unlocked(dataset_path,log):
    """Function to check for a lockfile in a given dataset before starting
    a reduction.  Returns True if unlocked."""

    lockfile = path.join(dataset_path,'dataset.lock')

    if path.isfile(lockfile) == True:
        log.info(' -> '+path.basename(dataset_path)+' is locked')
        status = False
    else:
        log.info(' -> '+path.basename(dataset_path)+' is not locked')
        status = True

    return status

def trigger_parallel_auto_reduction(config,dataset_dir,phot_db_path,data_status):
    """Function to spawn a child process to run the reduction of a
    single dataset.

    Inputs:
        setup       PipelineSetup object
        dataset_dir   str    Path to dataset red_dir
    """

    command = path.join(config['software_dir'],'reduction_control.py')
    args = [config['python_exec_path'], command, dataset_dir, phot_db_path, 'auto', data_status]

    p = subprocess.Popen(args, stdout=subprocess.PIPE)

    return p.pid

if __name__ == '__main__':
    run_pipeline()
