# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 09:07:37 2018

@author: rstreet
"""
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm
import os
from sys import argv
plt.rc('font', family='DejaVu Sans')
plt.rc('xtick', labelsize='large')
plt.rc('ytick', labelsize='large')


def map_DE_population(input_file):
    """Function to plot maps of the (logs .vs. logq) parameter space mapped
    out by a Differential Evolution (DE) algorithm search.

    The first plot is a 2D histogram of the number of DE samples
    for each pixel in (logs .vs. logq) space, as a proxy for a map of the
    population of solutions calculated during a Differential Evolution
    search of parameter space.

    The second plot is a 2D histogram of the chi squared values of each pixel
    in the parameter space.

    Based on original code by E. Bachelet.
    """

    map_data = np.loadtxt(input_file)
    map_data_sort = map_data[map_data[:,-1].argsort(),]

    fig = plt.figure(1,(5,5))

    # Down sample the number of map points to speed up plotting
    #index = np.random.randint(0,len(map_data),int(len(map_data)*0.12))
    index = np.arange(0,len(map_data),1, dtype=int)

    #plt.subplot(121)

    plt.hist2d(map_data_sort[index,4],map_data_sort[index,5],
               norm=LogNorm(),bins=(30,30))

    plt.title('N DE samples')
    plt.xlabel('$log_{10}(s)$')
    plt.ylabel('$log_{10}(q)$')
    plt.colorbar()

    #plt.subplot(122)
    plt.savefig('DE_population_map_nsamples.pdf',bbox_inches='tight')

    plt.close(1)

    # Down sample the number of map points to speed up plotting
    #index = np.random.randint(0,len(map_data),int(len(map_data)*0.12))
    index = np.arange(0,len(map_data),2, dtype=int)

    fig = plt.figure(2,(5,5))

    plt.scatter(map_data[index,4],map_data[index,5],
                c=np.log10(map_data[index,-1]),alpha=0.25, s=2)

    plt.title('$\log_{10}(\chi^{2})$')
    plt.xlabel('$log_{10}(s)$')
    plt.ylabel('$log_{10}(q)$')
    plt.colorbar()

    plt.savefig('DE_population_maps_chisq.pdf',bbox_inches='tight')

    plt.close(2)

    index = np.arange(0,len(map_data),1, dtype=int)

    fig = plt.figure(3,(5,5))

    n_bin_q = 30
    n_bin_s = 30

    chisq = np.log10(map_data[index,-1])

    logs_bins = np.arange(map_data_sort[index,4].min(),
                           map_data_sort[index,4].max(), n_bin_s)
    s_bin_incr = (map_data_sort[index,4].max()-map_data_sort[index,4].min())/n_bin_s

    logq_bins = np.arange(map_data_sort[index,5].min(),
                           map_data_sort[index,5].max(), n_bin_q)
    q_bin_incr = (map_data_sort[index,5].max()-map_data_sort[index,5].min())/n_bin_q

    chisq_image = np.zeros([len(logq_bins),len(logs_bins)])

    for s,s_min in enumerate(logs_bins):

        s_max = s_min + s_bin_incr

        idx1 = np.where(map_data_sort[index,4] >= s_min)[0]
        idx2 = np.where(map_data_sort[index,4] <= s_max)[0]

        sidx = list(set(idx2).intersection(set(idx2)))

        for q,q_min in enumerate(logq_bins):

            q_max = q_min + q_bin_incr

            idx1 = np.where(map_data_sort[index,5] >= q_min)[0]
            idx2 = np.where(map_data_sort[index,5] <= q_max)[0]
            qidx = list(set(idx2).intersection(set(idx2)))

            idx = list(set(sidx).intersection(set(qidx)))

            chisq_image[q,s] = np.median(chisq[idx])

    plt.imshow(chisq_image)

    plt.title('Median $\log_{10}(\chi^{2})$')
    plt.xlabel('$log_{10}(s)$')
    plt.ylabel('$log_{10}(q)$')
    plt.colorbar()

    #plt.subplot(122)
    plt.savefig('DE_population_map_chisq_median.pdf',bbox_inches='tight')

    plt.close(3)

if __name__ == '__main__':

    if len(argv) == 1:
        input_file = input('Please enter the path to the DE population output: ')

    else:
        input_file = argv[1]

    map_DE_population(input_file)
