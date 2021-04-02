from os import path, remove
from shutil import rmtree
from pyDANDIA import pipeline_setup
from pyDANDIA import logs
from sys import argv

def reset_stage5_data_products(setup,log):

    log.info('Removing any stage 5 data products from earlier reductions, if present')

    log_file = path.join(setup.red_dir, 'stage5.log')
    if path.isfile(log_file):
        remove(log_file)
        log.info(' -> Removed stage 5 log file')

    dir_list = [ path.join(setup.red_dir, 'diffim'),
                 path.join(setup.red_dir, 'kernel') ]
    for dir in dir_list:
        if path.isdir(dir):
            rmtree(dir)
            log.info(' -> Removed '+dir)


if __name__ == '__main__':

    setup = pipeline_setup.PipelineSetup()
    if len(argv) > 1:
        setup.red_dir = argv[1]
    else:
        setup.red_dir = input('Please enter the path to the reduction directory: ')

    log = logs.start_stage_log(setup.red_dir, 'reset_stage5')

    reset_stage5_data_products(setup,log)

    logs.close_log(log)
