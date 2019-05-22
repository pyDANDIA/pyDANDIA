# -*- coding: utf-8 -*-
"""
Created on Tue Jul 31 15:04:19 2018

@author: rstreet
"""

from os import path
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import jester_phot_transforms

def plot_3D_extinction_data():
    """Function to plot the 3D reddening derived from the maps from Pan-STARRS1
    by Green et al. 2015ApJ...810...25G"""
    
    if len(argv) == 1:
        
        data_file = input('Please enter the path to the data file: ')
        plot_file = input('Please enter the path to the output plot file: ')
        D_source = float(input('Please enter the distance of the source [kpc]: '))
        D_lens = float(input('Please enter the distance of the lens [kpc]: '))
        sig_D_lens = float(input('Please enter the uncertainty on the distance of the lens: '))
        
    else:
        
        data_file = argv[1]
        plot_file = argv[2]
        D_source = float(argv[3])
        D_lens = float(argv[4])
        sig_D_lens = float(argv[5])
        
    (DistMod,EBV) = read_3D_map_data(data_file)
    
    (DM_source, sig_DM_source) = calc_distance_modulus(D_source, 0.0)
    (DM_lens, sig_DM_lens) = calc_distance_modulus(D_lens, sig_D_lens)
    
    print('Distance modulus to: ')
    print('Source: '+str(DM_source)+' +/- '+str(sig_DM_source)+' mag')
    print('Lens: '+str(DM_lens)+' +/- '+str(sig_DM_lens)+' mag')
    
    (EBV_interp, EBV_lens, sig_EBV_lens, Av_lens, sig_Av_lens) = interpolate_colour_excess(DistMod, EBV, DM_lens, sig_DM_lens)
    
    print('Interpolated estimate of the extinction and reddening to the lens:')
    print('Av = '+str(Av_lens)+' +/- '+str(sig_Av_lens))
    print('E(B-V) = '+str(EBV_lens)+' +/- '+str(sig_EBV_lens))
    
    plot_EBV_distance(DistMod, EBV, DM_source, DM_lens, EBV_interp, plot_file)
    
    results = jester_phot_transforms.transform_JohnsonCousins_to_SDSS(BV=EBV_lens, sigBV=sig_EBV_lens)
    
    print('Results from Jester transforms:')
    print('E(g-r) = '+str(results['g-r'])+' +/- '+str(results['siggr']))
    
    calc_schlafly_reddening(EBV_lens,sig_EBV_lens)
    
def calc_distance_modulus(D, sig_D):
    """Function to calculate the distance modulus.
    Note: assumes input distance is in kiloparsecs    
    """
    
    DM = 5.0 * np.log10(D*1000.0) - 5.0
    
    sig_DM = (sig_D/D)*DM
    
    return DM, sig_DM
    
def read_3D_map_data(data_file):
    """Function to read the results of a query through
    http://argonaut.skymaps.info/
    """
    
    if path.isfile(data_file) == False:
        print('ERROR: Cannot find input file '+data_file)
        exit()
        
    line_list = open(data_file,'r').readlines()
    
    DistMod = []
    EBV = []
    
    for i in range(0,len(line_list),1):
        
        line = line_list[i]
        
        if line[0:1] != '#':
            
            if 'SampleNo | DistanceModulus' in line:
                
                columns = line_list[i+1]
                
                for item in columns.replace('|','').replace('\n','').split():
                    
                    if len(item) > 0:
                        DistMod.append(float(item))
            
                i += 1
            
            if 'BestFit' in line:
                
                columns = line.replace('|','').replace('\n','').split()
                
                for item in columns[1:]:
                    
                    if len(item) > 0:
                        EBV.append(float(item))
    
    DistMod = np.array(DistMod)
    EBV = np.array(EBV)
    
    return DistMod, EBV
    
def interpolate_colour_excess(DistMod, EBV, DM_lens, sig_DM_lens, 
                              Rv_GB=2.5, sig_Rv_GB=0.2):
    """Function to interpolate the colour excess curve and extract a 
    precise estimate for the reddening suffered by the lens.

    The resulting E(B-V)_lens value is used to estimate extinction, Av
    based on the relative visibility value for the Bulge determined by
    Nataf et al 2012, 2013, ApJ, 769, 88, Rv~2.5.
    """
    
    f = interp1d(DistMod, EBV)
    
    EBV_lens = f(DM_lens)
    
    min_ebv = f(DM_lens - sig_DM_lens)
    max_ebv = f(DM_lens + sig_DM_lens)
    
    sig_EBV_lens = (max_ebv - min_ebv) / 2.0
    
    Av = Rv_GB * EBV_lens
    
    sig_Av = np.sqrt( (sig_Rv_GB/Rv_GB)**2 + (sig_EBV_lens/EBV_lens)**2 ) * Av
    
    return f, EBV_lens, sig_EBV_lens, Av, sig_Av
    
def plot_EBV_distance(DistMod, EBV, DM_source, DM_lens, EBV_interp, 
                      plot_file, plot_interpolation=True):
    """Function to plot the reddening as a function of distance along the
    line of sight"""
    
    fig = plt.figure(1,(10,10))

    plt.rcParams.update({'font.size': 18})
    
    plt.plot(DistMod,EBV,'k-',alpha=0.2)
    
    [xmin,xmax,ymin,ymax] = plt.axis()
    
    if DM_source < xmax:
        ydata = np.arange(EBV.min(), EBV.max()+0.2, 0.5)    
        xdata = np.zeros(len(ydata))
        xdata.fill(DM_source)
    
        plt.plot(xdata, ydata,'m-.')

    else:
        
        plt.arrow(xmax-1.0, (ymax-ymin)/2.0, 1.0, 0.0)
        plt.text('$source_{m-M}$ = '+str(round(DM_source,1))+' mag', 
                 xmax-1.0, (ymax-ymin)/2.0)
                 
    ydata = np.arange(EBV.min(), EBV.max()+0.2, 0.5)    
    xdata = np.zeros(len(ydata))
    xdata.fill(DM_lens)

    plt.plot(xdata, ydata,'b--')
    
    if plot_interpolation:
        f = EBV_interp(DistMod)
        plt.plot(DistMod, f, 'k:')
        
    plt.xlabel('Distance Modulus [mag]')
    plt.ylabel('E(B-V) [mag]')

    plt.grid()
    
    plt.savefig(plot_file, bbox_inches='tight')

def interpolate_Rv_to_bandpass(Rv,bandpass):
    """Function to convert the reddening value provided in the V-band (Rv)
    to the bandpass indicated, using the data from Table 6 from 
    Schalafly & Finkbeiner, 2011, ApJ, 737, 103.
    """

    Rv_table_data = np.array( [ 2.1, 3.1, 4.1, 5.1 ] )
    
    table_data = { 'SDSS-g': [3.843, 3.303, 3.054, 2.910],
                   'SDSS-r': [2.255, 2.285, 2.300, 2.308],
                   'SDSS-i': [1.583, 1.698, 1.751, 1.782],
                   'SDSS-z': [1.211, 1.263, 1.286, 1.300] }    
    
    if Rv < Rv_table_data.min() or Rv > Rv_table_data.max():

        print('ERROR: Rv ='+str(Rv)+' lies outside the range of available data.  No estimation of reddening coefficient can be provided.')
        exit()

    if bandpass not in table_data.keys():
        
        print('ERROR: No data available for bandpass '+bandpass+'. No estimation of reddening coefficient can be provided.')
        exit()

    ydata = table_data[bandpass]

    f = interp1d(Rv_table_data, ydata)
    
    R_bandpass = f(Rv)
    
    return R_bandpass

def calc_schlafly_reddening(EBV_lens, sig_EBV_lens, Rv_GB=2.5, sig_Rv_GB=0.2):
    """Function to calculate the exinction in different bandpasses, based on 
    the data in Schalafly & Finkbeiner, 2011, ApJ, 737, 103.
    """
    
    print('Extinction and reddening from estimated from Schlafly & Finkbeiner data:')
    for b in ['SDSS-g', 'SDSS-r', 'SDSS-i']:
        Rb = interpolate_Rv_to_bandpass(Rv_GB,b)
        Ab = Rb * EBV_lens
        sig_Ab = (sig_EBV_lens/EBV_lens) * Ab
        print('A('+b+') = '+str(Ab)+' +/- '+str(sig_Ab)+', R('+b+') = '+str(Rb))


if __name__ == '__main__':

    plot_3D_extinction_data()