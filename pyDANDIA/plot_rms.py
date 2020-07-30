# -*- coding: utf-8 -*-
"""
@author: rstreet
"""

from sys import argv
from os import path
import numpy as np
from pyDANDIA import hd5_utils
from pyDANDIA import logs
from pyDANDIA import metadata
from pyDANDIA import pipeline_setup
import matplotlib.pyplot as plt

def calc_rms():

    params = get_args()

    log = logs.start_stage_log( params['red_dir'], 'rms' )

    reduction_metadata = metadata.MetaData()
    reduction_metadata.load_a_layer_from_file(params['red_dir'], 'pyDANDIA_metadata.fits', 'matched_stars')
    matched_stars = reduction_metadata.load_matched_stars()

    photometry_data = fetch_dataset_photometry(params,log)

    phot_statistics = calc_mean_rms_mag(photometry_data,log)
    print(phot_statistics)
    
    plot_rms(phot_statistics, params, log)

    logs.close_log(log)

def fetch_dataset_photometry(params,log):

    setup = pipeline_setup.pipeline_setup({'red_dir': params['red_dir']})

    photometry_data = hd5_utils.read_phot_hd5(setup)

    log.info('Loaded photometric data for '+str(len(photometry_data))+' stars')

    return photometry_data

def calc_mean_rms_mag(photometry_data,log):

    phot_statistics = np.zeros( (len(photometry_data),2) )

    (phot_statistics[:,0], _) = calc_weighted_mean_2D(photometry_data, 1, 2)
    log.info('Calculated stellar mean magnitudes weighted by the photometric uncertainties')

    phot_statistics[:,1] = calc_weighted_rms(photometry_data, phot_statistics[:,0], 1, 2)
    log.info('Calculated RMS per star weighted by the photometric uncertainties')

    return phot_statistics

def calc_weighted_mean_2D(data, col, errcol):

    wmean = np.zeros(len(data))
    werror = np.zeros(len(data))

    for j in range(0,len(data),1):
        mask = np.logical_and(data[j,:,col] > 0.0, data[j,:,errcol] > 0.0)

        err_squared_inv = 1.0 / (data[j,mask,errcol]*data[j,mask,errcol])

        wmean[j] =  (data[j,mask,col] * err_squared_inv).sum() / (err_squared_inv.sum())

        werror[j] = 1.0 / (err_squared_inv.sum())

    return wmean, werror

def calc_weighted_rms(data, mean_mag, magcol, errcol):

    rms = np.zeros(len(data))

    for j in range(0,len(data),1):
        mask = np.logical_and(data[j,:,magcol] > 0.0, data[j,:,errcol] > 0.0)

        err_squared_inv = 1.0 / (data[j,mask,errcol]*data[j,mask,errcol])

        rms[j] =  np.sqrt( ((data[j,mask,magcol]-mean_mag[j])**2 * err_squared_inv).sum() / (err_squared_inv.sum()) )

    return rms

def plot_rms(phot_statistics, params, log):

    fig = plt.figure(1,(10,10))

    plt.plot(phot_statistics[:,0], phot_statistics[:,1], 'k.')

    plt.xlabel('Weighted mean mag')
    plt.ylabel('RMS [mag]')

    plt.grid()

    plot_file = path.join(params['red_dir'],'rms.png')
    plt.savefig(plot_file)

    log.info('Output RMS plot to '+plot_file)

def get_args():

    params = {}

    if len(argv) == 1:

        params['db_file_path'] = input('Please enter the path to the field photometric DB: ')
        params['red_dir'] = input('Please enter the path to a dataset reduction directory: ')

    else:

        params['db_file_path'] = argv[1]
        params['red_dir'] = argv[2]

    return params

if __name__ == '__main__':
    calc_rms()
