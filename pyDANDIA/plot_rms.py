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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

def calc_rms():

    params = get_args()

    log = logs.start_stage_log( params['red_dir'], 'rms' )

    photometry_data = fetch_dataset_photometry(params,log)

    phot_statistics = calc_mean_rms_mag(photometry_data,log)

    plot_rms(phot_statistics, params, log)

    logs.close_log(log)

def fetch_dataset_photometry(params,log):

    setup = pipeline_setup.pipeline_setup({'red_dir': params['red_dir']})

    photometry_data = hd5_utils.read_phot_hd5(setup)

    log.info('Loaded photometric data for '+str(len(photometry_data))+' stars')

    return photometry_data

def calc_mean_rms_mag(photometry_data,log):

    phot_statistics = np.zeros( (len(photometry_data),3) )

    (phot_statistics[:,0], _) = calc_weighted_mean_2D(photometry_data, 11, 12)
    log.info('Calculated stellar mean magnitudes weighted by the photometric uncertainties')

    phot_statistics[:,1] = calc_weighted_rms(photometry_data, phot_statistics[:,0], 11, 12)
    log.info('Calculated RMS per star weighted by the photometric uncertainties')

    phot_statistics[:,2] = calc_percentile_rms(photometry_data, phot_statistics[:,0], 11, 12)
    log.info('Calculated RMS per star using percentile method')

    return phot_statistics

def calc_weighted_mean_2D(data, col, errcol):

    mask = np.invert(np.logical_and(data[:,:,col] > 0.0, data[:,:,errcol] > 0.0))
    mags = np.ma.array(data[:,:,col], mask=mask)
    errs = np.ma.array(data[:,:,errcol], mask=mask)

    err_squared_inv = 1.0 / (errs*errs)
    wmean =  (mags * err_squared_inv).sum(axis=1) / (err_squared_inv.sum(axis=1))
    werror = 1.0 / (err_squared_inv.sum(axis=1))

    return wmean, werror

def calc_weighted_rms(data, mean_mag, magcol, errcol):

    mask = np.invert(np.logical_and(data[:,:,magcol] > 0.0, data[:,:,errcol] > 0.0))
    mags = np.ma.array(data[:,:,magcol], mask=mask)
    errs = np.ma.array(data[:,:,errcol], mask=mask)

    err_squared_inv = 1.0 / (errs*errs)
    dmags = (mags.transpose() - mean_mag).transpose()
    rms =  np.sqrt( (dmags**2 * err_squared_inv).sum(axis=1) / (err_squared_inv.sum(axis=1)) )

    return rms

def calc_percentile_rms(data, mean_mag, magcol, errcol):

    mask = np.invert(np.logical_and(data[:,:,magcol] > 0.0, data[:,:,errcol] > 0.0))
    mags = np.ma.array(data[:,:,magcol], mask=mask)

    rms_per = (np.percentile(mags,84, axis=1)-np.percentile(mags,16, axis=1))/2

    return rms_per

def plot_rms(phot_statistics, params, log):

    fig = plt.figure(1,(10,10))
    plt.rcParams.update({'font.size': 18})

    mask = np.logical_and(phot_statistics[:,0] > 0.0, phot_statistics[:,1] > 0.0)
    plt.plot(phot_statistics[mask,0], phot_statistics[mask,1], 'k.',
            marker=".", markersize=0.5, alpha=0.5, label='Weighted RMS')

    mask = np.logical_and(phot_statistics[:,0] > 0.0, phot_statistics[:,2] > 0.0)
    plt.plot(phot_statistics[mask,0], phot_statistics[mask,2], 'g.',
                    marker="+", markersize=0.5, alpha=0.5, label='Percentile RMS')

    plt.yscale('log')
    plt.xlabel('Weighted mean mag')
    plt.ylabel('RMS [mag]')

    plt.grid()
    l = plt.legend()
    plt.tight_layout()

    [xmin,xmax,ymin,ymax] = plt.axis()
    plt.axis([xmin,xmax,1e-3,5.0])

    plot_file = path.join(params['red_dir'],'rms.png')
    plt.savefig(plot_file)

    log.info('Output RMS plot to '+plot_file)

def get_args():

    params = {}

    if len(argv) == 1:

        params['red_dir'] = input('Please enter the path to a dataset reduction directory: ')

    else:

        params['red_dir'] = argv[1]

    return params

if __name__ == '__main__':
    calc_rms()
