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

    phot_statistics = calc_mean_rms_mag(photometry_data,log,'calibrated')

    plot_rms(phot_statistics, params, log)

    logs.close_log(log)

def fetch_dataset_photometry(params,log):

    setup = pipeline_setup.pipeline_setup({'red_dir': params['red_dir']})

    photometry_data = hd5_utils.read_phot_hd5(setup)

    log.info('Loaded photometric data for '+str(len(photometry_data))+' stars')

    return photometry_data

def get_photometry_columns(phot_columns='instrumental'):
    """Function to return the column indices of the magnitude and magnitude uncertainties
    in the photometry array for a single dataset.  Options are:
    instrumental
    calibrated
    corrected
    """

    if phot_columns == 'instrumental':
        mag_col = 11
        merr_col = 12
    elif phot_columns == 'calibrated':
        mag_col = 13
        merr_col = 14
    elif phot_columns == 'corrected':
        mag_col = 23
        merr_col = 24

    return mag_col, merr_col

def calc_mean_rms_mag(photometry_data,log,phot_columns):

    (mag_col, merr_col) = get_photometry_columns(phot_columns)

    phot_statistics = np.zeros( (len(photometry_data),4) )

    (phot_statistics[:,0], phot_statistics[:,3]) = calc_weighted_mean_2D(photometry_data, mag_col, merr_col)
    log.info('Calculated stellar mean magnitudes weighted by the photometric uncertainties')

    phot_statistics[:,1] = calc_weighted_rms(photometry_data, phot_statistics[:,0], mag_col, merr_col)
    log.info('Calculated RMS per star weighted by the photometric uncertainties')

    phot_statistics[:,2] = calc_percentile_rms(photometry_data, phot_statistics[:,0], mag_col, merr_col)
    log.info('Calculated RMS per star using percentile method')

    return phot_statistics

def calc_weighted_mean_2D(data, col, errcol):

    mask = np.invert(np.logical_and(data[:,:,col] > 0.0, data[:,:,errcol] > 0.0))
    mags = np.ma.array(data[:,:,col], mask=mask)
    errs = np.ma.array(data[:,:,errcol], mask=mask)

    test_star = 148465
    print('MAGS: ',mags[test_star,:])
    print('ERRS: ',errs[test_star,:])
    idx = np.where(mags > 0.0)
    err_squared_inv = 1.0 / (errs*errs)
    wmean =  (mags * err_squared_inv).sum(axis=1) / (err_squared_inv.sum(axis=1))
    werror = np.sqrt( 1.0 / (err_squared_inv.sum(axis=1)) )
    print('WMEAN = ',wmean[test_star], werror[test_star])

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

def plot_rms(phot_statistics, params, log, plot_file=None):

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

    if plot_file == None:
        plot_file = path.join(params['red_dir'],'rms.png')
    plt.savefig(plot_file)

    log.info('Output RMS plot to '+plot_file)
    plt.close(1)

def output_phot_statistics(phot_statistics, file_path, log):

    f = open(file_path, 'w')
    f.write('# Star_index  weighted_mean_mag  weighted_rms percentile_rms weighted_mean_mag_error')
    for j in range(0,len(phot_statistics),1):
        f.write(str(j)+' '+str(phot_statistics[j,0])+' '+str(phot_statistics[j,1])+' '+\
                str(phot_statistics[j,2])+' '+str(phot_statistics[j,3])+'\n')
    f.close()
    log.info('Output photometric statistics to '+file_path)

def get_args():

    params = {}

    if len(argv) == 1:

        params['red_dir'] = input('Please enter the path to a dataset reduction directory: ')

    else:

        params['red_dir'] = argv[1]

    return params

if __name__ == '__main__':
    calc_rms()
