import numpy as np
import os
import sys
cwd = os.getcwd()
sys.path.append(os.path.join(cwd,'../'))
import photometry_classes
import analyse_cmd

def test_load_target_timeseries_photometry():

    if len(os.argv) < 4:
        g_filec = input('Please enter path to the g-band lightcurve file: ')
        r_file = input('Please enter path to the g-band lightcurve file: ')
        i_file = input('Please enter path to the g-band lightcurve file: ')
    else:
        g_file = os.argv[1]
        r_file = os.argv[2]
        i_file = os.argv[3]
        
    load_target_timeseries_photometry(config,photometry,log)


if __name__ == '__main__':

    test_load_target_timeseries_photometry()
