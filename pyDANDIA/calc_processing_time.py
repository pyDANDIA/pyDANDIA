from os import path
from sys import argv
from datetime import datetime, timedelta

def reduction_processing_time(red_dir):

    log_list = ['stage0.log', 'stage1.log', 'stage2.log', 'reference_astrometry.log',
                'stage3.log', 'stage3_db_ingest.log', 'stage4.log', 'stage5.log', 'stage6.log']

    total_time = timedelta(seconds=0.0)

    for log in log_list:
        log_file = path.join(red_dir, log)

        if path.isfile(log_file):
            (d1, d2) = get_log_start_and_end_date(log_file)
            dt = d2 - d1
            print(log+'  '+str(dt))

            total_time += dt

    print('Total processing time: '+str(total_time))

def get_log_start_and_end_date(log_file):

    lines = open(log_file, 'r').readlines()

    d1 = datetime.strptime(lines[0].split()[0], '%Y-%m-%dT%H:%M:%S')

    i = -1
    d2 = None
    while d2 == None:
        if len(lines[i].replace('\n','')) > 0:
            d2 = datetime.strptime(lines[i].split()[0], '%Y-%m-%dT%H:%M:%S')
        else:
            i -= 1

    return d1, d2

if __name__ == '__main__':

    if len(argv) == 1:
        red_dir = input('Please enter the path to the reduction directory: ')
    else:
        red_dir = argv[1]

    reduction_processing_time(red_dir)
