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
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import argparse

def calc_field_cmds():

    params = get_args()

    log = logs.start_stage_log( params['red_dir'], 'field_cmd' )

    # Crossmatch table provides information on the filter used for each image
    xmatch = crossmatch.CrossMatchTable()
    xmatch.load(params['crossmatch_file'],log=log)
    log.info('Loaded crossmatch table for the field')

    # Select stars with valid photometry in both passbands
    col1 = params['blue_column']
    col2 = params['red_column']
    jdx1 = np.logical_and(xmatch.stars[col1] > 0.0, xmatch.stars[col2] > 0.0)
    select_stars = np.where(jdx1)[0]
    log.info(str(len(select_stars))+' stars have valid measurements in both bands')

    # Extract the photometry data for all stars in the field
    data = np.zeros((len(select_stars),3))
    data[:,0] = xmatch.stars['field_id'][select_stars]
    data[:,1] = xmatch.stars[col1][select_stars]
    data[:,2] = xmatch.stars[col1][select_stars] - xmatch.stars[col2][select_stars]

    # If a specific target was indicated, extract the photometry for that
    # object
    if params['target_field_id']:
        field_idx = params['target_field_id'] - 1
        target_mag = xmatch.stars[col1][field_idx]
        target_colour = xmatch.stars[col1][field_idx] - xmatch.stars[col2][field_idx]
        target_params = [params['target_field_id'], target_mag, target_colour]
    else:
        target_params = [None, None, None]

    # Plot interactive RMS diagram
    plot_file = path.join(params['red_dir'], params['plot_file'])
    axis_labels = [params['xlabel'], params['ylabel']]
    plotly_lightcurves.plot_interactive(data, plot_file, axis_labels,
                target_params,
                title=params['field_name']+' CMD', xreverse=True)

    logs.close_log(log)

def get_args():

    parser = argparse.ArgumentParser(prog='plot_field_cmd')
    parser.add_argument('red_dir', help='The path to the top-level data directory')
    parser.add_argument('field_name', help='Name of the field')
    parser.add_argument('blue_column', help='Xmatch.stars table column name of the bluewards passband')
    parser.add_argument('red_column', help='Xmatch.stars table column name of the redwards passband')
    parser.add_argument('xlabel', help='Label for the CMD x-axis')
    parser.add_argument('ylabel', help='Label for the CMD y-axis')
    parser.add_argument('target_field_id', help='To highlight a specific object, enter its field ID or None')
    parser.add_argument('plot_file_name', help='Name of the output plotfile')

    args = parser.parse_args()

    params = {'red_dir': args.red_dir,
              'field_name': args.field_name,
              'blue_column': args.blue_column,
              'red_column': args.red_column,
              'xlabel': args.xlabel,
              'ylabel': args.ylabel,
              'target_field_id': args.target_field_id,
              'plot_file': args.plot_file_name}
    params['crossmatch_file'] = path.join(params['red_dir'],
                            params['field_name']+'_field_crossmatch.fits')
    if 'none' in str(params['target_field_id']).lower():
        params['target_field_id'] = None
    else:
        params['target_field_id'] = int(params['target_field_id'])

    return params


if __name__ == '__main__':
    calc_field_cmds()
