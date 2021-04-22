import numpy as np
from os import path
from sys import argv
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import string
import glob

def inspect_resampled_params(params):

    (frame_list, coefficients) = load_resampled_data(params)

    plot_coefficients(params, frame_list, coefficients)

    filter_coefficients(params, frame_list, coefficients)

def load_resampled_data(params):

    resampled_dir = path.join(params['red_dir'], 'resampled')
    if not path.isdir(resampled_dir):
        raise IOError('Cannot find the directory for resampled data at '+resampled_dir)

    frame_list = glob.glob(path.join(resampled_dir, '*.fits'))
    if len(frame_list) == 0:
        raise IOError('No stage 4 output per frame available')

    coefficients = np.zeros((9,len(frame_list)))
    for i,frame in enumerate(frame_list):
        f = path.join(resampled_dir, frame, 'warp_matrice_stamp_'+str(params['stamp_number'])+'.npy')

        if path.isfile(f):
            coefficients[:,i] = np.load(f).flatten()

    frames = []
    for f in frame_list:
        frames.append(path.basename(f))

    return frames, coefficients

def plot_coefficients(params, frame_list, coefficients):

    col_keys = list(mcolors.TABLEAU_COLORS.keys())
    index = np.arange(0,9,1)
    markers = ['o', 'v', '3', 's', 'P', '*', 'X', 'D', '+']

    frame_index = np.arange(0,coefficients.shape[1],1)

    fig = plt.figure(1,(10,10))
    for i in index:
        plt.plot(frame_index, coefficients[i,:], marker=markers[i],
                markerfacecolor=mcolors.TABLEAU_COLORS[col_keys[i]],
                markeredgecolor=mcolors.TABLEAU_COLORS[col_keys[i]],
                label=string.ascii_lowercase[i])
    plt.xlabel('Frame')
    plt.ylabel('Coefficient value')
    plt.xticks(frame_index, frame_list, rotation=45.0, ha='right')
    plt.legend()
    plt.tight_layout()

    plt.savefig(path.join(params['red_dir'], 'warp_matrices.png'), )

def filter_coefficients(params, frame_list, coefficients):

    threshold = 100.0

    idx = np.where(coefficients > threshold)[0]

    if len(idx) == 0:
        print('No frames exceed QC threshold of '+str(threshold))
    else:
        print('The following frames exceed the QC threshold of '+str(threshold))
        for i in idx:
            print(frame_list[i])

if __name__ == '__main__':

    params = {}
    if len(argv) == 1:
        params['red_dir'] = input('Please enter the path to the reduction directory: ')
        params['stamp_number'] = int(input('Please enter the stamp number to inspect: '))
    else:
        params['red_dir'] = argv[1]
        params['stamp_number'] = int(argv[2])

    inspect_resampled_params(params)
