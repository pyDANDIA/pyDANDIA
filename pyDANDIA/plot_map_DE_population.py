# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 09:07:37 2018

@author: rstreet
"""

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
    
    fig = plt.figure(1,(10,5))
        
    # Down sample the number of map points to speed up plotting
    #index = np.random.randint(0,len(map_data),int(len(map_data)*0.12))
    index = np.arange(0,len(map_data),1, dtype=int)
    
    plt.subplot(121)
    
    plt.hist2d(map_data_sort[index,4],map_data_sort[index,5],
               norm=LogNorm(),bins=(30,30))
    
    plt.title('N DE samples')
    plt.xlabel('$log_{10}(s)$')
    plt.ylabel('$log_{10}(q)$')
    plt.colorbar()

    plt.subplot(122)

    plt.scatter(map_data[index,4],map_data[index,5],
                c=np.log10(map_data[index,-1]),alpha=0.25)
    
    plt.title('$\log_{10}(\chi^{2})$')
    plt.xlabel('$log_{10}(s)$')
    plt.ylabel('$log_{10}(q)$')
    plt.colorbar()

    plt.savefig('DE_population_maps.eps',bbox_inches='tight')
    
    plt.close(1)
    
if __name__ == '__main__':
    
    if len(argv) == 1:
        input_file = input('Please enter the path to the DE population output: ')
        
    else:
        input_file = argv[1]
    
    map_DE_population(input_file)
    