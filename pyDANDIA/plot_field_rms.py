from sys import argv
from os import path
import numpy as np
from pyDANDIA import hd5_utils
from pyDANDIA import logs
from pyDANDIA import metadata
from pyDANDIA import pipeline_setup
from pyDANDIA import normalize_photometry
from pyDANDIA import crossmatch
from pyDANDIA import field_photometry
from pyDANDIA import plot_rms
from pyDANDIA import plotly_lightcurves
from pyDANDIA import config_utils
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse

def calc_field_rms():

    config = get_args()

    log = logs.start_stage_log( config['red_dir'], 'field_rms' )

    # Crossmatch table provides information on the filter used for each image
    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(path.join(config['red_dir'],config['field_name']+'_field_crossmatch.fits'),log=log)
    log.info('Loaded crossmatch table for the field')

    # Exclude stars with poor photometry
    qc_mask = select_by_photometry_quality(xmatch, config)

    filter_list = np.unique(xmatch.images['filter'].data)
    log.info('Identified list of filters to process: '+repr(filter_list))

    log.info('Loading the timeseries photometry...')
    phot_data = hd5_utils.read_phot_from_hd5_file(path.join(config['red_dir'],
                                                            config['field_name']+'_quad'+config['quadrant']\
                                                            +'_photometry.hdf5'),
                                                  return_type='array')
    log.info('-> Completed photometry load')

    # By default, select the columns of photometry that have been normalized:
    (mag_col, mag_err_col) = field_photometry.get_field_photometry_columns('normalized')
    qc_col = 16

    # Plot a separate RMS diagram for each filter
    for filter in filter_list:
        image_index = np.where(xmatch.images['filter'] == filter)[0]
        phot_data_filter = phot_data[:,image_index,:]
        jdx = np.where(xmatch.field_index['quadrant'] == float(int(config['quadrant'])))[0]
        log.info(str(len(jdx))+' stars in quadrant '+config['quadrant'])

        # Calculate lightcurve statistics, filtering out any with zero measurements:
        phot_statistics = np.zeros( (len(phot_data_filter),3) )

        phot_statistics[:,0] = xmatch.stars['field_id'][jdx]
        (phot_statistics[:,1],werror) = plot_rms.calc_weighted_mean_2D(phot_data_filter, mag_col, mag_err_col, qc_col=qc_col)
        phot_statistics[:,2] = plot_rms.calc_weighted_rms(phot_data_filter, phot_statistics[:,1], mag_col, mag_err_col, qc_col=qc_col)

        selection = np.logical_and(phot_statistics[:,1] > 0.0, phot_statistics[:,2] > 0.0)
        selection = np.logical_and(selection, qc_mask)
        phot_statistics = phot_statistics[selection]

        # Plot interactive RMS diagram
        axis_labels = ['Mag', 'RMS [mag]']
        target_params = [False]
        plot_title = 'RMS diagram for '+config['field_name']+', quadrant '+config['quadrant']+', '+filter+'-band'
        if config['interactive']:
            plot_file = path.join(config['output_dir'],
                                  config['field_name']+'_quad'+config['quadrant']+'_rms_postnorm'+'_'+filter+'.html')
            plotly_lightcurves.plot_interactive(phot_statistics, plot_file, axis_labels,
                    target_params, title=plot_title, logy=True, xreverse=True)
        else:
            plot_file = path.join(config['output_dir'],
                                  config['field_name']+'_quad'+config['quadrant']+'_rms_postnorm'+'_'+filter+'.png')
            plot_static_rms(config, phot_statistics, filter, log, plot_file=plot_file)

    logs.close_log(log)

def select_by_photometry_quality(xmatch, config):
    gcol = 'cal_g_mag_'+config['reference_dataset_code']
    gerrcol = 'cal_g_magerr_'+config['reference_dataset_code']
    rcol = 'cal_r_mag_'+config['reference_dataset_code']
    rerrcol = 'cal_r_magerr_'+config['reference_dataset_code']
    icol = 'cal_i_mag_'+config['reference_dataset_code']
    ierrcol = 'cal_i_magerr_'+config['reference_dataset_code']

    qc_mask = np.logical_and(xmatch.stars[gcol] > 0.0, xmatch.stars[gerrcol] <= config['g_sigma_max'])
    qc_mask = np.logical_and(qc_mask, xmatch.stars[rcol] > 0.0)
    qc_mask = np.logical_and(qc_mask, xmatch.stars[rerrcol] <= config['r_sigma_max'])
    qc_mask = np.logical_and(qc_mask, xmatch.stars[icol] >  0.0)
    qc_mask = np.logical_and(qc_mask, xmatch.stars[ierrcol] <= config['i_sigma_max'])

    if len(qc_mask) == 0:
        raise ValueError('All stars excluded by quality control criteria')

    return qc_mask


def plot_static_rms(config, phot_statistics, filter, log, plot_file=None):

    fig = plt.figure(1,(10,10))
    plt.rcParams.update({'font.size': 18})

    mask = np.logical_and(phot_statistics[:,1] > 0.0, phot_statistics[:,2] > 0.0)
    plt.plot(phot_statistics[mask,1], phot_statistics[mask,2], 'k.',
            marker=".", markersize=0.5, alpha=0.5, label='Weighted RMS')

    plt.yscale('log')
    plt.xlabel('Weighted mean mag')
    plt.ylabel('RMS [mag]')

    plt.grid()
    l = plt.legend()
    plt.tight_layout()

    [xmin,xmax,ymin,ymax] = plt.axis()
    xmin = config['plot_'+filter.replace('p','')+'_range'][0]
    xmax = config['plot_'+filter.replace('p','')+'_range'][1]
    ymin = config['plot_rms_range'][0]
    ymax = config['plot_rms_range'][1]
    plt.axis([xmin,xmax,ymin,ymax])

    if plot_file == None:
        plot_file = path.join(config['output_dir'],'rms.png')
    plt.savefig(plot_file)

    log.info('Output RMS plot to '+plot_file)
    plt.close(1)

def get_args():

    parser = argparse.ArgumentParser()
    parser.add_argument('config_file', help='Path to the plot configuration file')
    args = parser.parse_args()

    config = config_utils.build_config_from_json(args.config_file)

    for key in ['interactive']:
        if 'false' in str(config[key]).lower():
            config[key] = False
        else:
            config[key] = True

    return config


if __name__ == '__main__':
    calc_field_rms()
