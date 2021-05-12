from os import path
from sys import argv
from astropy.io import fits
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def diff_images_qc(params):

    diff_dir = path.join(params['red_dir'],'diffim')
    diff_images = glob.glob(path.join(diff_dir, '*'))

    dimage_stats = []
    for dimage_path in diff_images:
        stats = calc_stamp_statistics(params,dimage_path)
        dimage_stats.append(stats)
    dimage_stats = np.array(dimage_stats)

    plot_dimage_statistics(params, dimage_stats, diff_images)

def calc_stamp_statistics(params,dimage_path):
    statistics = []

    if params['stamp_number'] == -1:
        stamps = glob.glob(path.join(dimage_path,'diff_stamp_*.fits'))

        for i,stamp in enumerate(stamps):
            image = fits.getdata(stamp)
            statistics.append([i,np.median(image), image.std()])
    else:
        stamp = path.join(dimage_path,'diff_stamp_'+str(params['stamp_number'])+'.fits')
        image = fits.getdata(stamp)
        statistics.append([params['stamp_number'],np.median(image), image.std()])

    return statistics

def plot_dimage_statistics(params,dimage_stats,diff_images):

    markers = ['.', 'v', 's', 'p', '*', '+', 'X', 'd', '1', '3', 'D', '^', 'P', '>', '<', '4']
    col_keys = list(mcolors.TABLEAU_COLORS.keys()) + list(mcolors.TABLEAU_COLORS.keys())

    dimage_index = np.arange(0,len(diff_images),1)
    frames = []
    for f in diff_images:
        frames.append(path.basename(f))

    (fig, (ax0, ax1)) = plt.subplots(nrows=2, ncols=1)
    fig.set_size_inches(40, 10)

    if params['stamp_number'] == -1:
        stamp_index = np.arange(0,dimage_stats.shape[1],1)
    else:
        stamp_index = np.array([params['stamp_number']])

    for stamp_idx in range(0,dimage_stats.shape[1],1):
        ax0.plot(dimage_index, dimage_stats[:,stamp_idx,1], marker=markers[stamp_idx],
                    markerfacecolor=mcolors.TABLEAU_COLORS[col_keys[stamp_idx]],
                    markeredgecolor=mcolors.TABLEAU_COLORS[col_keys[stamp_idx]])
        ax0.set(xlabel='Image', ylabel='Mean pixel value [ADU]')

        ax1.plot(dimage_index, dimage_stats[:,stamp_idx,2], marker=markers[stamp_idx],
                    markerfacecolor=mcolors.TABLEAU_COLORS[col_keys[stamp_idx]],
                    markeredgecolor=mcolors.TABLEAU_COLORS[col_keys[stamp_idx]])
        ax1.set(xlabel='Image', ylabel='Std. Dev [ADU]')

    for ax in [ax0, ax1]:
        ax.grid()
        ax.set_xticks(dimage_index)
        ax.set_xticklabels(frames)
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",rotation_mode="anchor")

    plt.tight_layout()
    plt.savefig(path.join(params['red_dir'], 'diff_image_statistics.png'), )

if __name__ == '__main__':
    params = {}
    if len(argv) == 1:
        params['red_dir'] = input('Please enter the path to the reduction directory: ')
        params['stamp_number'] = int(input('Please enter the stamp number to inspect or -1 for all: '))
    else:
        params['red_dir'] = argv[1]
        params['stamp_number'] = int(argv[2])

    diff_images_qc(params)
