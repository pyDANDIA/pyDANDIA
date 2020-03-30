from os import path
from sys import argv
import matplotlib.pyplot as plt
import numpy as np
from astropy import table

# Necessary to ensure that the plt.show call brings up
# an interactive window by default:
plt.get_current_fig_manager().show()

def get_args():

    params = {}
    if len(argv) == 1:
        params['data_file_path'] = input('Please enter path to the input datafile: ')
        params['plot_file_path'] = input('Please enter path to the output plotfile: ')
    else:
        params['data_file_path'] = argv[1]
        params['plot_file_path'] = argv[2]


    if len(argv) > 3:
        for a in argv[3:]:
            params[a] = True

    return params

def load_lc(params):

    if not path.isfile(params['data_file_path']):
        raise IOError('Cannot find lightcurve file '+params['data_file_path'])

    raw_data = np.loadtxt(params['data_file_path'])

    mask = [(raw_data[:,1] > 0.0) & (raw_data[:,2] < 0.1)]

    mag_col = 1
    merr_col = 2
    if '--cal' in params.keys() or '--calibrated' in params.keys():
        mag_col = 3
        merr_col = 4

    data = table.Table( [table.Column(name='hjd', data=raw_data[mask][:,0]-2450000.0),
                         table.Column(name='mag', data=raw_data[mask][:,mag_col]),
                         table.Column(name='mag_err', data=raw_data[mask][:,merr_col])] )
    return data

def plot_lc(params):

    data = load_lc(params)

    fig = plt.figure(1,(10,10))

    plt.errorbar(data['hjd'], data['mag'], yerr=data['mag_err'],
                    mfc='black', mec='red', marker='.', fmt='none')

    plt.xlabel('HJD-2450000.0')
    if '--cal' in params.keys() or '--calibrated' in params.keys():
        plt.ylabel('Calibrated magnitude')
    else:
        plt.ylabel('Magnitude')

    [xmin, xmax, ymin, ymax] = plt.axis()
    plt.axis([xmin, xmax, ymax, ymin])

    if '--i' in params.keys() or '--interactive' in params.keys():
        plt.show()
    else:
        plt.savefig(params['plot_file_path'])

    plt.close()

if __name__ == '__main__':

    params = get_args()

    plot_lc(params)
